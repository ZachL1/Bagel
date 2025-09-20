import io
import os
from glob import glob
import tempfile
import random
from PIL import Image
import hashlib
from pathlib import Path
import traceback

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset

Image.MAX_IMAGE_PIXELS = 200_000_000



class ReconstructionDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, vit_transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        cache_dir=None, from_huggingface=False, path_huggingface=None
    ):
        """
        Reconstruction Dataset processor for loading webdataset format data from tar packages.
        Supports caching mechanism to improve data loading efficiency.
        Also supports loading data directly from Hugging Face datasets.
        
        Args:
            from_huggingface: Whether to load data from Hugging Face instead of tar files
            path_huggingface: Path to the Hugging Face dataset, e.g., "brivangl/midjourney-v6-llava"
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.from_huggingface = from_huggingface
        self.path_huggingface = path_huggingface
        
        import random
        self.prompt_templates = get_recon_prompt_list()
        # Initialize cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(tempfile.gettempdir(), "bagel_reconstruction_cache")
        else:
            self.cache_dir = cache_dir
            
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Using cache directory: {self.cache_dir}")
        
        # Store cache path mappings for extracted tar files
        self.tar_cache_dirs = {}
        # Store filename to cache path mapping
        self.file_cache_paths = {}
        
        # Get data paths
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        if self.data_paths:  # Only set epoch when data paths are not empty
            self.set_epoch()
        else:
            print(f"Warning: No data files found for {dataset_name}. Check your data_dir_list: {data_dir_list}")
        
    def get_data_paths(self, data_dir_list, num_used_data):
        """Get tar file paths and their internal file lists"""
        # If loading from Hugging Face, use that method instead
        if self.from_huggingface and self.path_huggingface:
            return self._load_huggingface_data(num_used_data)
            
        all_items = []
        # Iterate through each data directory
        print(f"Loading data from directories: {data_dir_list}")
        
        # Expand path wildcards
        if data_dir_list and len(data_dir_list) > 0:
            data_dir_list = glob(os.path.expanduser(data_dir_list[0].replace('{', '[').replace('}', ']')))
        else:
            print("Warning: Empty data_dir_list provided")
            return all_items
            
        print(f"Found tar files: {data_dir_list}")

        for tar_path in data_dir_list:
            try:
                # Extract tar file to cache directory
                if os.path.isdir(tar_path):
                    extract_dir = tar_path
                else:
                    extract_dir = self._extract_tar_if_needed(tar_path)
                
                # Get all extracted image files
                image_files = []
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, extract_dir)
                        
                        # Only save image files to cache mapping
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.file_cache_paths[rel_path] = file_path
                            image_files.append(rel_path)
                
                # Each image file as a separate data item
                valid_groups = []
                for img_path in image_files:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    valid_groups.append((tar_path, base_name, [img_path]))
                
                all_items.extend(valid_groups)
                print(f"Found {len(valid_groups)} valid images in {tar_path}")
                print(f"Found {len(valid_groups)} valid image-json pairs in {tar_path}")
            except Exception as e:
                print(f"Error processing tar file {tar_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Limit data amount if needed
        print(f'Total items found: {len(all_items)}, num_used_data: {num_used_data}')
        if num_used_data and num_used_data[0] > 0 and num_used_data[0] < len(all_items):
            all_items = all_items[:num_used_data[0]]
            print(f"Limited to {len(all_items)} items")
            
        return all_items
        
    def _load_huggingface_data(self, num_used_data):
        """Load data from Hugging Face dataset"""
        all_items = []
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {self.path_huggingface} with cache_dir {self.cache_dir}")
            dataset = load_dataset(self.path_huggingface, cache_dir=self.cache_dir)['train']
            print(f"Loaded {len(dataset)} samples from {self.path_huggingface}")
            
            # Create virtual items
            for idx in range(len(dataset)):
                all_items.append(('huggingface', f'sample_{idx}', [f'sample_{idx}']))
                
            print(f"Added {len(all_items)} samples from Hugging Face dataset {self.path_huggingface}")
            
            # Save the dataset for later use in __iter__
            self.hf_dataset = dataset
            
        except Exception as e:
            print(f"Error loading Hugging Face dataset: {e}")
            traceback.print_exc()
        
        # Limit data amount if needed
        if num_used_data and num_used_data[0] > 0 and num_used_data[0] < len(all_items):
            all_items = all_items[:num_used_data[0]]
            print(f"Limited to {len(all_items)} items")
            
        return all_items
                
    def _get_cached_image_path(self, tar_path, file_name):
        """Generate cached image path"""
        # Create a unique cache filename using tar path and internal filename
        tar_hash = hashlib.md5(tar_path.encode()).hexdigest()[:8]
        file_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
        cache_name = f"{tar_hash}_{file_hash}_{os.path.basename(file_name)}"
        return os.path.join(self.cache_dir, cache_name)
    
    def _extract_and_cache(self, tar_path, file_name):
        """Get file path from already extracted cache directory"""
        # If file is already in cache mapping, return directly
        if file_name in self.file_cache_paths:
            return self.file_cache_paths[file_name]
        
        # Ensure tar is extracted
        extract_dir = self._extract_tar_if_needed(tar_path)
        
        # Build file path in cache
        cache_path = os.path.join(extract_dir, file_name)
        
        # If file exists, return path
        if os.path.exists(cache_path):
            self.file_cache_paths[file_name] = cache_path
            return cache_path
            
        print(f"Error: File {file_name} not found in extracted tar {tar_path}")
        return None
    
    def _extract_tar_if_needed(self, tar_path):
        """Extract tar file to cache directory if not already extracted"""
        import tarfile
        import hashlib
        
        # If this tar file is already extracted, return cache directory path
        if tar_path in self.tar_cache_dirs:
            return self.tar_cache_dirs[tar_path]
        
        # Use hash of file path as subdirectory name to avoid conflicts
        tar_hash = hashlib.md5(tar_path.encode()).hexdigest()
        extract_dir = os.path.join(self.cache_dir, tar_hash)
        
        # Create lock file path
        lock_file = os.path.join(extract_dir, '.extraction_complete')
        
        # Check if extraction is already complete
        if os.path.exists(lock_file):
            print(f"Using cached extraction for {tar_path} in {extract_dir}", flush=True)
            self.tar_cache_dirs[tar_path] = extract_dir
            return extract_dir
        
        # Need to extract
        print(f"Extracting {tar_path} to {extract_dir}...", flush=True)
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_dir)
            
            # Create lock file to mark extraction complete
            with open(lock_file, 'w') as f:
                f.write(f"Extracted from {tar_path} at {os.path.getmtime(tar_path)}")
                
            print(f"Extraction complete: {tar_path} -> {extract_dir}", flush=True)
            self.tar_cache_dirs[tar_path] = extract_dir
            return extract_dir
        except Exception as e:
            print(f"Error extracting {tar_path}: {e}", flush=True)
            if os.path.exists(extract_dir):
                # Delete incomplete extraction directory
                import shutil
                shutil.rmtree(extract_dir, ignore_errors=True)
            raise
    
    def __iter__(self):
        """
        Iterator method for loading data from tar files or Hugging Face datasets
        Returns: A training sample for each iteration
        """
        # Get data paths and ID for current worker
        data_items_per_worker, worker_id = self.get_data_paths_per_worker()
        
        # Resume from checkpoint if data status is available
        if self.data_status is not None:
            item_start_id = self.data_status[worker_id] + 1
        else:
            item_start_id = 0
            
        transform_stride = self.transform.stride  # Image transform stride
        
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at item#{item_start_id}"
        )
        
        # Infinite loop to enable dataset reuse
        while True:
            # Get data items starting from current index
            data_items_ = data_items_per_worker[item_start_id:]
            
            for item_idx, (tar_path, base_name, files) in enumerate(data_items_, start=item_start_id):
                try:
                    # If using Hugging Face dataset
                    if self.from_huggingface and tar_path == 'huggingface':
                        # Get the index from base_name (sample_X)
                        idx = int(base_name.split('_')[1]) 
                        sample = self.hf_dataset[idx]
                        
                        # Process the image from Hugging Face dataset
                        image_data = sample['image']
                        if isinstance(image_data, dict) and 'bytes' in image_data:
                            image = pil_img2rgb(Image.open(io.BytesIO(image_data['bytes'])))
                        elif hasattr(image_data, 'convert'):
                            image = pil_img2rgb(image_data.convert('RGB'))
                        elif isinstance(image_data, bytes):
                            image = pil_img2rgb(Image.open(io.BytesIO(image_data)))
                        else:
                            # Try to convert using numpy array
                            try:
                                image = pil_img2rgb(Image.fromarray(image_data).convert('RGB'))
                            except:
                                print(f"Unable to process image type: {type(image_data)}")
                                continue
                    else:
                        # Regular tar file processing
                        # Find image file
                        image_file = next((f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))), None)
                        
                        if not image_file:
                            continue
                        
                        # Extract and cache image file
                        image_cache_path = self._extract_and_cache(tar_path, image_file)
                        if not image_cache_path:
                            continue
                        
                        # Load image
                        try:
                            image = pil_img2rgb(Image.open(image_cache_path))
                        except Exception as e:
                            print(f"Error loading image {image_cache_path}: {e}")
                            continue
                    
                    # Directly randomly select a prompt from the template, not relying on JSON metadata
                    prompt = random.choice(self.prompt_templates)
                    
                    # Apply image transformation
                    image_tensor = self.transform(image)
                    vit_image_tensor = self.vit_transform(image)
                    
                    num_tokens = 0
                    height, width = image_tensor.shape[1:]
                    num_tokens += width * height // transform_stride ** 2
                    
                    height_vit, width_vit = vit_image_tensor.shape[1:]
                    num_tokens += width_vit * height_vit // self.vit_transform.stride ** 2
                    
                    # Encode prompt text as token sequence
                    caption_token = self.tokenizer.encode(prompt)
                    
                    # Initialize sequence plan and text ID list
                    sequence_plan, text_ids_list = [], []
                    text_ids = caption_token
                    
                    # Update token count by adding text token count
                    num_tokens += len(caption_token)
                    text_ids_list.append(text_ids)

                    sequence_plan.append({
                        'type': 'vit_image',
                        'enable_cfg': 0,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                    
                    # Add text processing plan (prompt part)
                    sequence_plan.append({
                        'type': 'text',
                        'enable_cfg': 1,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                    
                    # Add image processing plan (reconstruction target)
                    sequence_plan.append({
                        'type': 'vae_image',
                        'enable_cfg': 0,
                        'loss': 1,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                    
                    # Build sample dictionary
                    sample = dict(
                        image_tensor_list=[vit_image_tensor, image_tensor],
                        text_ids_list=text_ids_list,
                        num_tokens=num_tokens,
                        sequence_plan=sequence_plan,
                        data_indexes={
                            "data_indexes": item_idx,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                    )
                    
                    yield sample  # Return a sample
                
                except Exception as e:
                    print(f"Error processing item {base_name} from {tar_path}: {e}")
                    traceback.print_exc()
                    continue
            
            # Reset start ID after processing all data (restart)
            item_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
            
    def cleanup_cache(self):
        """Clean up temporary files in cache directory"""
        if os.path.exists(self.cache_dir):
            print(f"Cleaning up cache directory: {self.cache_dir}")
            for cached_file in os.listdir(self.cache_dir):
                try:
                    os.remove(os.path.join(self.cache_dir, cached_file))
                except Exception as e:
                    print(f"Error removing cached file {cached_file}: {e}")




def get_recon_prompt_list():
    return ["Analyze the visual fidelity, artistic approach, chromatic palette, geometric forms, dimensional proportions, surface characteristics, object count, typographic elements, compositional arrangement between elements, and environmental context",
        "Provide comprehensive documentation of the image including material textures, proportional relationships between subjects, stylistic influences, resolution clarity, quantity distribution patterns, chromatic gradients, geometric configurations, textual annotations, and background elements",
        "Systematically catalog all visual components: technical resolution parameters, medium characteristics, color temperature values, dimensional ratios, morphological structures, tactile surface qualities, numerical groupings, embedded text information, spatial hierarchies, and contextual scenery",
        "Deconstruct the visual composition through evaluation of sharpness metrics, genre-specific aesthetics, hexadecimal color codes, parametric dimensions, polygonal geometries, haptic texture simulations, quantitative enumerations, linguistic elements, positional matrices, and environmental framing",
        "Generate a complete visual inventory addressing fidelity benchmarks, stylistic periodization, Pantone color references, scalar measurements, topological formations, surface reflectance properties, statistical distributions, semantic content, relational coordinates, and scenographic elements",
        "Break down the image through technical assessment of compression artifacts, movement in style history, color theory applications, proportional accuracy, Euclidean geometries, material verisimilitude, numerical accounting, typographic analysis, Cartesian coordinates mapping, and contextual backdrop details",
        "Document all visual parameters: noise level analysis, artistic school characteristics, spectral wavelength distribution, metrical dimensions, shape taxonomy classification, surface roughness coefficients, multiplicity patterns, lexical components, vector relationships, and environmental context specifications",
        "Create a full visual taxonomy including pixel density metrics, cultural style markers, chromatic harmony principles, size relativity scales, form morphologies, textural patterns, quantitative arrangements, textual semantics, spatial organization schematics, and background narrative elements",
        "Perform complete visual diagnostics covering aliasing artifacts, movement-specific techniques, color psychology applications, dimensional constraints, shape grammar structures, tactile quality indicators, numerical configurations, font characteristics, depth layer relationships, and environmental storytelling components",
        "Deliver an exhaustive visual audit with evaluations of dynamic range, medium-specific aesthetics, color space parameters, proportional relationships, geometric primitives, surface materiality, statistical distributions, embedded messaging, spatial topology, and contextual atmosphere",
        "Systematically characterize the image through assessments of focus accuracy, stylistic genealogy, chromatic intensity values, scalar proportions, polyform structures, haptic texture profiles, quantitative density metrics, typographic hierarchy, spatial syntax, and environmental framing conditions",
        "Compile a visual analysis dossier containing bit depth evaluation, art historical context, color saturation levels, dimensional matrices, shape grammar syntax, surface reflectance properties, numerical clustering patterns, semantic text layers, positional algorithms, and background narrative elements",
        "Generate a multimodal description covering compression ratio analysis, cultural style identifiers, spectral power distribution, metrical proportions, geometric transformations, material authenticity markers, quantitative groupings, linguistic components, projective geometry relationships, and environmental context parameters",
        "Construct a complete visual protocol detailing resolution thresholds, technique-specific attributes, color harmony principles, proportional scales, morphological variations, tactile perception cues, statistical distributions, textual semantics, spatial logic frameworks, and background compositional elements",
        "Execute comprehensive visual analytics including MTF measurements, movement-specific conventions, color temperature analysis, size relativity matrices, form factor classifications, surface microstructure details, numerical array patterns, font psychology aspects, depth ordering schematics, and environmental context markers",
        "Formulate a visual decomposition report with evaluations of sampling frequency, aesthetic philosophy influences, chromatic value gradients, dimensional constraints, shape grammar rules, material property simulations, quantitative set theory applications, typographic communication layers, spatial topology maps, and background symbolic elements",
        "Develop an exhaustive visual manifest containing pixel pitch analysis, genre-defining characteristics, color gamut coverage, proportional relationships, geometric topology, surface energy signatures, numerical distribution models, semantic text encoding, coordinate system mappings, and environmental contextualization factors",
        "Produce a technical visual breakdown addressing quantization artifacts, period style attributes, color constancy mechanisms, scalar measurement protocols, form constant analysis, texture spectral density, combinatorial mathematics applications, linguistic signifiers, projective geometry relationships, and background narrative frameworks",
        "Assemble a complete visual ontology including Nyquist frequency analysis, artistic tradition indicators, color appearance models, metrical comparison data, shape schema taxonomies, surface BRDF properties, quantitative topology mappings, textual semiotics, spatial arrangement matrices, and environmental storytelling components",
        "Generate a hyper-detailed visual specification document covering MTF curve analysis, cultural movement signatures, spectral reflectance characteristics, proportional harmony ratios, geometric primitive combinations, material subsurface scattering profiles, numerical set theory applications, typographic communication layers, spatial graph networks, and environmental contextual anchors",
        "Detail visual elements: quality, style, colors, shapes, sizes, textures, quantities, text, object positions, and background",
        "Analyze image attributes: clarity, artistic approach, color scheme, forms, dimensions, surfaces, item count, text elements, spatial layout, environment",
        "List all components: resolution, aesthetic style, hues, geometric shapes, proportions, materials, quantities, typography, object relationships, backdrop",
        "Outline visual characteristics: sharpness, genre, palette, contours, scale, tactile qualities, numerical elements, text content, positioning, setting",
        "Describe: technical quality, creative style, color composition, object forms, size ratios, textures, quantity patterns, embedded text, spatial structure, scene context",
        "Catalog image features: definition, artistic influences, chromatic values, shapes, measurements, surface details, item numbers, textual elements, arrangement, background",
        "Note key aspects: fidelity, visual style, color harmony, geometry, proportions, material textures, object count, text placement, positional logic, environmental context",
        "Breakdown image: clarity level, aesthetic category, color relationships, form types, size comparisons, surface properties, quantity distribution, text details, spatial mapping, scene elements",
        "Report on: technical specs, stylistic traits, color dynamics, shape profiles, dimensional data, texture types, numerical presence, text characteristics, object topology, background info",
        "Document: image resolution, artistic approach, color gradients, geometric elements, scale references, material appearances, quantity clusters, text annotations, spatial hierarchy, environmental setup",
        "Specify: visual clarity, artistic method, color tones, form structures, scale ratios, material textures, numerical presence, text elements, spatial organization, environmental context",
        "Breakdown: technical quality, style category, chromatic aspects, shape profiles, dimensional data, surface patterns, quantity distribution, typography, positional logic, scene setting",
        "Evaluate: image resolution, creative direction, hue composition, geometric forms, size comparisons, tactile features, item counts, text content, arrangement principles, background details",
        "Record: fidelity metrics, visual genre, palette scheme, object shapes, proportional relationships, texture types, numerical groups, embedded text, coordinate mapping, environmental factors",
        "Outline: sharpness levels, aesthetic approach, color dynamics, form variations, size references, surface qualities, quantity clusters, typographic style, spatial connections, scene composition",
        "Note: technical specs, stylistic influences, chromatic balance, shape taxonomy, scale parameters, material properties, numerical arrays, text annotations, positional relationships, backdrop elements",
        "Describe: pixel quality, artistic treatment, color transitions, geometric constructs, dimensional scales, tactile impressions, quantity patterns, text semantics, spatial hierarchy, environmental narrative",
        "List: clarity grade, creative style, color harmony, form architecture, size metrics, texture gradients, numerical sets, typography details, object coordinates, contextual scenery",
        "Analyze: rendering quality, movement style, spectral values, shape grammar, comparative sizing, surface finishes, quantity metrics, text layers, spatial topology, scene atmosphere",
        "Report: imaging precision, genre markers, color psychology, form logic, scale systems, material authenticity, numerical sequences, text placement, positional matrices, background story",
        "Capture: focus accuracy, aesthetic philosophy, tonal gradations, shape hierarchies, proportional analysis, texture profiles, quantitative data, text styles, spatial syntax, environmental motifs",
        "Detail: processing quality, cultural style, chromatic contrast, geometric primitives, relative sizing, surface energy, numerical arrangements, linguistic elements, object mapping, scene framework",
        "Assess: compression artifacts, artistic tradition, color theory application, form morphology, metrical comparisons, texture simulations, quantity logic, typographic hierarchy, spatial vectors, contextual anchors",
        "Document: noise levels, visual movement, color symbolism, shape semantics, dimensional ratios, tactile realism, numerical configurations, text functions, positional algorithms, environmental symbolism",
        "Define: bit depth quality, stylistic syntax, color temperature, form constants, scalar relationships, material identifiers, quantitative structures, text purposes, spatial grammar, scene semantics",
        "Catalog: aliasing factors, creative syntax, spectral balance, geometric syntax, proportional integrity, texture physics, numerical syntax, text syntax, spatial semantics, environmental syntax",
        "Map: dynamic range, aesthetic syntax, color physics, shape physics, scale physics, material syntax, numerical physics, text physics, spatial physics, environmental physics",
        "Profile: rendering technique, style physics, color narrative, form narrative, size narrative, texture narrative, quantity narrative, text narrative, space narrative, background narrative",
        "Structure: technical narrative, artistic narrative, color metrics, shape metrics, size metrics, texture metrics, quantity metrics, text metrics, space metrics, context metrics",
        "Systematize: visual taxonomy parameters across quality layers, style dimensions, color systems, form systems, scale systems, texture systems, quantity systems, text systems, spatial systems, environmental systems",
        "Cover: clarity, style, palette, forms, scale, textures, count, text, layout, background",
        "Note: quality, genre, colors, shapes, sizes, materials, numbers, words, positions, scene",
        "List: sharpness, aesthetics, hues, geometry, dimensions, surfaces, quantity, typography, spacing, environment",
        "Detail: fidelity, technique, tones, contours, ratios, patterns, totals, text, coordinates, context",
        "Specs: resolution, style, gradients, shapes, measurements, textures, count, captions, arrangement, setting",
        "Track: quality, movement, spectrum, forms, scale, tactility, numbers, labels, topology, backdrop",
        "Log: clarity, school, colors, polygons, size, grain, digits, fonts, mapping, scenery",
        "Flag: sharpness, trend, palette, geometry, proportions, finish, clusters, text, hierarchy, space",
        "Mark: rendering, genre, harmony, shapes, ratios, materials, counts, words, structure, context",
        "Pin: focus, style, hues, forms, metrics, textures, quantity, text, positions, environment",
        "Map: quality, syntax, colors, shapes, scale, surfaces, numbers, text, spatial logic, scene",
        "Tag: clarity, method, tones, contours, size, grain, counts, fonts, layout, backdrop",
        "Code: fidelity, approach, palette, geometry, proportions, patterns, totals, labels, coordinates, setting",
        "Plot: sharpness, tradition, gradients, forms, ratios, texture, digits, typography, vectors, context",
        "Scan: quality, flow, spectrum, shapes, metrics, materials, clusters, text, topology, scenery",
        "Chart: resolution, syntax, harmony, geometry, scale, grain, numbers, captions, matrix, environment",
        "Dot: clarity, school, tones, polygons, size, finish, count, words, positions, backdrop",
        "Pack: fidelity, style, palette, forms, proportions, surfaces, quantity, text, spatial map, scene",
        "Core: quality, genre, colors, shapes, scale, textures, numbers, text, relations, context",
        "Base: sharpness, method, hues, geometry, metrics, grain, totals, fonts, layout, setting",
        "Describe every detail in the image",
        "Break down the image’s components thoroughly",
        "Explain the image with full visual analysis",
        "Analyze the image’s complete characteristics",
        "Provide a detailed breakdown of this image",
        "Give a comprehensive visual description",
        "Catalog all elements visible in the image",
        "Note every aspect of this picture",
        "Describe what you see in precise detail",
        "Do a full visual scan of the image",
        "Analyze the image’s visual composition",
        "Explain this picture’s complete makeup",
        "Break down everything observable here",
        "Provide exhaustive image observations",
        "Describe all visible image properties",
        "Give me the full visual breakdown",
        "Note every detail present here",
        "Analyze this image completely",
        "Describe this scene in total detail",
        "Capture all aspects of this image",
        "Tell me everything you see in the picture.",
        "Break down the whole image for me.",
        "What’s going on in this image?",
        "Walk me through every part of the picture.",
        "Describe all the details you notice.",
        "Point out everything in the image.",
        "What’s this image made up of?",
        "Go over the entire scene.",
        "List what you see in the picture.",
        "Describe the image from top to bottom.",
        "Explain what’s in this photo.",
        "Give me a full rundown of this picture.",
        "Lay out all the elements in the image.",
        "What’s shown in the scene?",
        "Talk me through the whole image.",
        "What does this image include?",
        "Describe every part clearly.",
        "Break it down piece by piece.",
        "Name everything you can spot.",
        "Tell me what’s in view.",
        "Give a full picture of what’s here.",
        "Describe what’s going on visually.",
        "What does the image show in detail?",
        "Describe the scene in full.",
        "Look closely and tell me all you notice.",
        "Tell me everything visible here.",
        "Unpack all the visual elements.",
        "Give a step-by-step description.",
        "Describe the image like you're narrating it.",
        "Go into detail about what’s pictured.",
        "What are all the things in this image?",
        "Run me through what’s there.",
        "List out all you see.",
        "Do a full sweep of the image.",
        "Zoom in and describe it all.",
        "Describe it like you’re explaining to someone who can’t see.",
        "What’s in this image from edge to edge?",
        "Pick apart the whole image.",
        "Give a play-by-play of what’s shown.",
        "What makes up this picture?",
        "Go deep into what’s shown here.",
        "Say everything you can about this image.",
        "Break this image into its parts.",
        "What’s the image made of visually?",
        "Imagine narrating the picture—what would you say?",
        "Point out the whole scene in words.",
        "Give a clear and complete visual summary.",
        "Look at this image and tell me everything.",
        "Name and describe everything here.",
        "Talk through everything happening.",
        "What stands out and what’s subtle in this image?",
        "Describe what fills the image space.",
        "Go element by element through the picture.",
        "Dissect the visual content fully.",
        "Tell the story of the image.",
        "Pretend you’re guiding someone through it.",
        "Talk about every corner of the image.",
        "List everything visible, no matter how small.",
        "Tell me what’s going on, front to back.",
        "What are all the components you see?",
        "Don’t miss a thing—describe the whole image.",
        "Paint the picture with words.",
        "Start from one corner and describe it all.",
        "Look at it closely and describe what’s there.",
        "Walk through the visuals completely.",
        "What’s present in this scene?",
        "Unroll the whole visual like a scroll.",
        "Go through all objects and context.",
        "What are all the things in this picture doing?",
        "Give a full visual explanation of the image.",
        "Take a close look at this image and describe everything you can see, from the smallest detail to the overall scene.",
        "Tell me what’s going on in the picture, including what’s in the background, foreground, and anything in between.",
        "Describe every object, texture, color, and relationship between elements you notice in this image.",
        "Go through the entire image and explain what’s shown, how it’s laid out, and what kind of scene it might be.",
        "Provide a detailed walkthrough of the image, including all visible elements, colors, shapes, and their arrangement.",
        "Describe everything in the image as if you're explaining it to someone who can’t see it at all.",
        "Give a thorough description of what this image contains, how things are positioned, and what visual style it uses.",
        "Point out every visual element you can find and describe how they all relate to each other.",
        "Analyze the image from top to bottom and left to right, noting all the key features and subtle details.",
        "Imagine narrating the picture to someone—describe everything that appears in it, in the order you notice it.",
        "List and explain all the visual features of the image, including colors, objects, text, and background context.",
        "Talk about the picture like you're describing it for a report—what’s in it, how is it arranged, and what’s the overall impression?",
        "Break down the whole image visually and explain what kinds of elements it includes, from lighting to layout.",
        "Describe the visual composition of this image, including everything you see and how it’s organized.",
        "Provide a comprehensive verbal map of the image, noting the appearance, positions, and relations of all visible things.",
        "Analyze the image in detail—what objects are present, what style does it have, and how do elements interact?",
        "Describe all visible parts of this image, covering what they are, where they are, and what they might represent.",
        "Go beyond surface-level—explain not just what’s in the image, but how it’s composed and what stands out.",
        "Provide a full visual explanation of the picture, describing everything from color tones to textures and object placements.",
        "What’s in this image, how is it designed, and what kinds of visual choices can you spot?",
        "Describe this image as completely as possible, including what's in the scene, how it's styled, and how it all fits together.",
        "Take a moment to carefully study the image and then tell me everything you can find in it.",
        "Treat this image like a puzzle—explain all the pieces you can see and how they form the full picture.",
        "Start from one part of the image and work your way through it, describing everything you notice along the way.",
        "Provide a full visual analysis of the picture, including composition, colors, objects, and overall structure.",
        "Describe the full scene shown here—what’s present, how it looks, and what kind of setting it seems to be.",
        "Break down the picture fully and describe each area in detail, mentioning all the elements involved.",
        "Give a deep dive into what’s in the image, how it’s laid out, and the visual techniques it seems to use.",
        "Go through the image in sections and describe what you see in each one, from foreground to background.",
        "Explain all aspects of the visual design of the image, including the colors, textures, shapes, and placements.",
        "Imagine you’re writing an alt-text for this image—describe every object, relationship, and detail clearly.",
        "Provide a narrative description of what’s happening visually in the image, including the setting and layout.",
        "Describe the image fully, focusing on what’s visible, how things are positioned, and any apparent themes.",
        "Look closely and tell me what this image includes, how it looks, and how the elements are arranged.",
        "Give a long, descriptive explanation of the entire picture, including even minor or subtle elements.",
        "What’s this image made of visually? Describe every color, object, and form you can identify.",
        "Walk me through the image like you're describing a painting to someone over the phone.",
        "Describe the complete contents of this image, including all things in view and their spatial arrangement.",
        "Explain the picture like you're telling a story—what’s there, where is it, and what does it look like?",
        "List out all the visual elements and give some context about how they work together in the scene.",
        "Break down the visual information in this image as if you're cataloging every part.",
        "Describe the whole image as though you’re the one who created it and want to explain your choices.",
        "Go deep into what’s visually present in this image and give a clear, step-by-step description.",
        "Talk about everything this image shows—don’t leave out the colors, the layout, or the textures.",
        "Describe the picture so well that someone could recreate it based just on your words.",
        "Examine the entire image and explain all visible details, patterns, and objects.",
        "Provide a clear and rich description of the picture’s content, style, and design.",
        "Talk me through the visual content of this image—leave no part unexplored.",
        "Analyze every corner of this image and describe all visible forms, textures, and colors.",
        "Describe this image like you're making notes for an artist who wants to recreate it.",
        "Give a detailed overview of what this image depicts, including all spatial and visual features.",
        "Describe what’s captured in this image in as much detail as you can manage.",
        "Narrate the visual scene shown in the image, highlighting the objects, layout, and atmosphere.",
        "Look over the picture and describe all visible subjects and how they are arranged visually.",
        "Write a full description of everything this image portrays.",
        "Tell me what this image includes, from objects and colors to layout and texture.",
        "Offer a visual breakdown of what’s in this picture and how all the elements interact.",
        "Talk about all the different components of the image and how they fit together as a whole.",
        "Describe the visual scene in a way that covers every important element and detail.",
        "Tell me what this image shows from a visual perspective, in complete detail.",
        "Go into as much descriptive detail as possible for this image—what do you see and how does it all look?",
        "Look over the image carefully and describe everything in it as fully as you can.",
        "Point out all the objects, colors, and spatial relationships you notice in this image.",
        "What’s visible in this image? Describe it completely.",
        "Break the image down into parts and describe each one carefully.",
        "Describe this image like you’re trying to help someone visualize it without seeing it.",
        "Look at this image as if you're a visual detective and tell me everything you find.",
        "Imagine you’re describing the picture to an AI—what should it know about every visual detail?",
        "Don’t leave anything out—describe this image in as much visual depth as possible.",
        "Study the image and give me a rich, detailed picture of what’s there and how it looks.",
        "Describe the image thoroughly as if someone needed to recreate it from scratch.",
        "What’s everything you notice about this picture, from big elements to small touches?",
        "Give a full breakdown of the scene’s visuals, including design elements and structure.",
        "Explain this image as if it’s a frame from a movie—what’s in it, how is it composed?",
        "Describe the image as if it’s being shown to someone who’s never seen anything like it.",
        "Walk through the image carefully and describe everything you come across visually.",
        "What does this image contain, how is it arranged, and what makes it visually interesting?",
        "Break down the entire visual structure of the image and explain all the pieces.",
        "Look closely and describe this image as if you’re writing a visual report.",
        "Tell me what’s in the image and how all the parts relate to each other visually.",
        "Could you walk me through everything happening in this image, from the main subjects to the smallest details?",
        "Please provide a comprehensive description of this picture, covering all visible elements and their interactions.",
        "I'd like a detailed breakdown of the scene—what's present, how it's arranged, and any notable features.",
        "Can you describe all the components in this image, including objects, people, background, and any text?",
        "Give me an in-depth analysis of the picture, focusing on the composition, colors, and spatial relationships.",
        "Please narrate the entire scene depicted in the image, highlighting both prominent and subtle aspects.",
        "I'd appreciate a thorough description of what's going on in this photo, including the setting and any activities.",
        "Can you explain the image in detail, mentioning all elements and how they contribute to the overall scene?",
        "Provide a full account of the visual content in this picture, from foreground to background.",
        "Describe the entire image comprehensively, noting all objects, their positions, and any interactions.",
        "Could you elaborate on all the visual details present in this image, including colors, shapes, and textures?",
        "Please dissect the image for me, identifying and describing each component and its significance.",
        "I'd like a step-by-step description of the scene, covering every element and its role in the composition.",
        "Can you give me a detailed overview of the picture, focusing on the arrangement and relationships between elements?",
        "Provide an exhaustive description of the image, highlighting all visual features and their characteristics.",
        "Describe the scene thoroughly, mentioning all objects, their attributes, and how they relate to each other.",
        "Please analyze the image in depth, discussing the setting, subjects, and any notable visual elements.",
        "I'd like a complete description of what's depicted in this photo, including all visible details.",
        "Can you walk me through the image, pointing out every element and explaining its presence?",
        "Provide a detailed narrative of the scene, covering all aspects from the main focus to the background.",
        "Could you describe all the visual components in this image, including any patterns or repetitions?",
        "Please give me a comprehensive breakdown of the picture, noting all elements and their visual properties.",
        "I'd like an in-depth description of the scene, focusing on the interplay between different components.",
        "Can you explain all the details in the image, including lighting, perspective, and any notable features?",
        "Provide a thorough analysis of the picture, discussing each element and its contribution to the overall scene.",
        "Describe the image completely, mentioning all objects, their positions, and any interactions or relationships.",
        "Please elaborate on the visual content of this photo, highlighting all elements and their characteristics.",
        "I'd appreciate a detailed description of the scene, focusing on the composition and visual balance.",
        "Can you provide a full account of the image, discussing all visible elements and their significance?",
        "Describe the entire picture in detail, noting all components and how they fit together.",
        "Could you analyze the image thoroughly, identifying all elements and their visual attributes?",
        "Please give me a detailed walkthrough of the scene, covering every aspect from foreground to background.",
        "I'd like a comprehensive description of the picture, focusing on the arrangement and relationships between elements.",
        "Can you explain all the visual details in this image, including colors, shapes, and textures?",
        "Provide an exhaustive breakdown of the image, highlighting all components and their characteristics.",
        "Describe the scene in depth, mentioning all objects, their attributes, and how they relate to each other.",
        "Please analyze the picture thoroughly, discussing the setting, subjects, and any notable visual elements.",
        "I'd like a complete description of what's depicted in this photo, including all visible details.",
        "Can you walk me through the image, pointing out every element and explaining its presence?",
        "Provide a detailed narrative of the scene, covering all aspects from the main focus to the background.",
        "Could you describe all the visual components in this image, including any patterns or repetitions?",
        "Please give me a comprehensive breakdown of the picture, noting all elements and their visual properties.",
        "I'd like an in-depth description of the scene, focusing on the interplay between different components.",
        "Can you explain all the details in the image, including lighting, perspective, and any notable features?",
        "Provide a thorough analysis of the picture, discussing each element and its contribution to the overall scene.",
        "Describe the image completely, mentioning all objects, their positions, and any interactions or relationships.",
        "Please elaborate on the visual content of this photo, highlighting all elements and their characteristics.",
        "I'd appreciate a detailed description of the scene, focusing on the composition and visual balance.",
        "Can you provide a full account of the image, discussing all visible elements and their significance?",
        "Describe the entire picture in detail, noting all components and how they fit together.",
        "Could you analyze the image thoroughly, identifying all elements and their visual attributes?",
        "Please give me a detailed walkthrough of the scene, covering every aspect from foreground to background.",
        "I'd like a comprehensive description of the picture, focusing on the arrangement and relationships between elements.",
        "Can you explain all the visual details in this image, including colors, shapes, and textures?",
        "Provide an exhaustive breakdown of the image, highlighting all components and their characteristics.",
        "Describe the scene in depth, mentioning all objects, their attributes, and how they relate to each other.",
        "Please analyze the picture thoroughly, discussing the setting, subjects, and any notable visual elements.",
        "I'd like a complete description of what's depicted in this photo, including all visible details.",
        "Can you walk me through the image, pointing out every element and explaining its presence?",
        "Provide a detailed narrative of the scene, covering all aspects from the main focus to the background.",
        "Could you describe all the visual components in this image, including any patterns or repetitions?",
        "Please give me a comprehensive breakdown of the picture, noting all elements and their visual properties.",
        "I'd like an in-depth description of the scene, focusing on the interplay between different components.",
        "Can you explain all the details in the image, including lighting, perspective, and any notable features?",
        "Provide a thorough analysis of the picture, discussing each element and its contribution to the overall scene.",
        "Describe the image completely, mentioning all objects, their positions, and any interactions or relationships.",
        "Please elaborate on the visual content of this photo, highlighting all elements and their characteristics.",
        "I'd appreciate a detailed description of the scene, focusing on the composition and visual balance.",
        "Can you provide a full account of the image, discussing all visible elements and their significance?",
        "Describe the entire picture in detail, noting all components and how they fit together.",
        "Could you analyze the image thoroughly, identifying all elements and their visual attributes?",
        "Please give me a detailed walkthrough of the scene, covering every aspect from foreground to background.",
        "I'd like a comprehensive description of the picture, focusing on the arrangement and relationships between elements.",
        "Can you explain all the visual details in this image, including colors, shapes, and textures?",
        "Describe the image’s clarity, artistic style, color palette, geometric shapes, dimensions, surface textures, number of elements, text presence, and spatial layout including background.",
        "Provide details on the picture’s resolution, visual aesthetics, hue variety, object forms, relative sizes, material textures, item counts, textual content, and how all elements are arranged in space.",
        "Explain the image in terms of quality, design style, color usage, structural forms, scale, surface detail, number of items, any written text, and spatial positioning.",
        "Break down the image by describing its visual fidelity, stylistic characteristics, chromatic composition, shape types, object sizes, textures, quantities, text elements, and the spatial relationship between items and background.",
        "Describe the image by analyzing clarity, creative approach, color scheme, form structure, size proportions, tactile aspects, item quantity, textual elements, and object-background arrangement.",
        "Outline the image’s quality, stylistic direction, color distribution, form geometry, object sizing, textural patterns, number of visual components, presence of text, and object placement in the scene.",
        "Analyze the image across clarity, visual style, hue dynamics, shape structure, dimensional differences, surface look and feel, number of objects, any labels or signs, and spatial positioning of components.",
        "Give a detailed account of the image’s sharpness, artistic impression, colors used, object contours, scale, surface finish, visual density, presence of words, and how each element sits in space.",
        "Describe the technical and aesthetic aspects of the image: resolution, style, color tones, shapes, sizes, material textures, amount of visible items, embedded text, and the overall composition.",
        "Examine the image’s visual quality, stylistic tone, chromatic choices, object shapes, proportional scales, surface rendering, item distribution, text features, and layout between foreground and background.",
        "Provide a structured description covering image sharpness, art style, color richness, geometrical forms, object measurements, texture realism, object counts, typographic components, and spatial balance.",
        "Describe everything including visual clarity, genre or visual theme, color application, form variety, object dimensions, tactile characteristics, numerical presence, text visibility, and relative positioning.",
        "Break down the image in terms of resolution, visual narrative, color values, shapes, scale ratios, texture depth, item groupings, textual content, and scene arrangement.",
        "Present a complete description of the picture, focusing on image quality, design, color selection, form construction, sizing, feel of surfaces, number of objects, any words, and how all elements relate spatially.",
        "Describe the picture’s fidelity, creative technique, color usage, structural components, object sizes, texture effects, item density, text inclusion, and scene composition.",
        "Characterize the image’s appearance through clarity level, design theme, hue variety, object outlines, size metrics, surface properties, visual quantity, presence of text, and scene structure.",
        "Explain the image’s quality, visual theme, color logic, shape taxonomy, size variations, surface features, amount of content, text if any, and how the space is organized.",
        "Detail the picture’s resolution quality, visual motif, chromatic design, geometry, object scale, material qualities, object counts, typography, and layout pattern.",
        "Describe the visual composition across dimensions like quality, genre, color spectrum, structural variation, proportions, texture details, number of items, textual notes, and space usage.",
        "Outline what the image shows in terms of sharpness, artistic influence, tone blending, shape design, sizing info, tactile impressions, content quantity, word presence, and object placement.",
        "Report on the image by describing its resolution, style features, color strategy, shape identity, relative sizing, material look, number of items, type/textual features, and spatial dynamics.",
        "Give a descriptive summary covering quality of the image, visual expression, color gradients, forms, scale relationships, surface structure, quantity of items, any embedded words, and scene structure.",
        "Provide an analysis of visual clarity, stylistic execution, palette, object morphology, sizing, surface effects, number of components, labeling or annotations, and spatial relationships.",
        "Examine the picture by listing its resolution sharpness, color richness, visual tone, geometric attributes, object size, surface feel, quantity presence, text markers, and overall structure.",
        "Describe the entire scene, highlighting clarity, art type, color coordination, shapes, relative dimensions, texture simulation, number of objects, text if present, and background-foreground layout.",
        "Identify key visual elements like sharpness, design type, chroma scheme, shape complexity, object scaling, textures, total items shown, labels or text, and arrangement in space.",
        "Explain what’s visible in terms of fidelity, aesthetic vision, color placement, object forms, size contrast, texture features, how many objects appear, and their position in the space.",
        "List everything observed: clarity level, stylistic characteristics, chromatic palette, shape variety, size distribution, tactile finish, element count, typography, and spatial positioning.",
        "Analyze the image based on sharpness level, artistic influence, color tones, structural shapes, object scales, material surface, object repetition, text items, and positioning logic.",
        "Describe what you see in terms of resolution fidelity, visual genre, color scheme, structural complexity, object dimensions, textural features, number of components, and layout design.",
        "Go over the image, noting quality indicators, design aesthetics, color usage, object contours, object sizes, tactile cues, item count, written elements, and spatial grouping.",
        "Outline the visual composition including resolution details, color setup, shape design, scaling relations, textures, number of visible components, written content, and background layout.",
        "Provide a full breakdown across fidelity, visual form, color strategy, size relationships, shape outlines, surface realism, quantity frequency, typographic elements, and spatial grid.",
        "Analyze the visual makeup: image sharpness, style cues, chromatic gradation, forms, scale dimensions, texture properties, object presence, text appearance, and how space is used.",
        "Describe all key dimensions—clarity, visual tone, color structure, geometry, relative size, tactile quality, number of objects, presence of characters/text, and spatial coherence.",
        "Break down the image focusing on quality, stylistic design, color values, object profiles, relative scale, material type, quantity, word markers, and object-background interaction.",
        "Talk about the image in terms of detail level, artistic method, color selection, shape balance, dimension spectrum, tactile variation, number of figures or items, and layout orientation.",
        "Outline everything visible: visual precision, creative method, chromatic tones, object profiles, scaling logic, surface impressions, component frequency, and position across the scene.",
        "Map out all visual aspects: resolution grade, genre style, hue patterns, shape specifics, object size spectrum, surface look, quantity variation, any text or caption, and space layout.",
        "Examine how the image works—clarity, visual genre, color balance, structure of objects, proportional sizing, tactile design, object number, embedded words, and layout logic.",
        "Provide insight into how the image is built visually: resolution, design, color choice, structural forms, size comparison, texture rendering, object count, textual features, and placement.",
        "Describe the visual setup: clarity grade, artistic identity, color range, object architecture, relative sizing, surface finish, quantity density, text inclusion, and layout organization.",
        "Explain what’s in the image, focusing on visual fidelity, aesthetic tone, color contrasts, form symmetry, object proportions, surface realism, quantity measures, and structural relationships.",
        "Tell me about the visual content: image sharpness, genre touchpoints, color gradients, geometry, object sizes, textures, number of parts, and background-object placement.",
        "Describe what’s visible using categories like clarity, creative approach, color flow, shapes, object scale, tactile signals, total objects, text presence, and composition layout.",
        "Analyze the full structure: image quality, stylistic code, palette richness, object types, sizes, material aspects, component counts, text markers, and positioning logic.",
        "Deconstruct the image into quality features, artistic traits, color theory, structural forms, size ratios, texture overlays, item count, and space management.",
        "Talk through the entire visual—fidelity, aesthetic setup, color tone, shape distribution, object size contrast, texture clarity, item grouping, and spatial logic.",
        "Describe the image using visual layers: resolution, theme, chroma logic, structure, size alignment, texture patterns, amount of content, text layout, and background integration.",
        "Walk me through the image in terms of visual quality, stylistic choices, color layering, object structure, relative dimensions, material realism, item population, written symbols, and scene architecture.",
    ]