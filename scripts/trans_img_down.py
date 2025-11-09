#!/usr/bin/env python3
"""
Script to download images from .jsonl file with resume capability and duplicate handling.
Supports multi-threaded downloading for improved performance.
"""

import json
import os
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests
from tqdm import tqdm
import argparse


def get_url_hash(url):
    """Generate a unique hash for a URL to use as filename."""
    return hashlib.md5(url.encode()).hexdigest()


def get_file_extension(url, default='.webp'):
    """Extract file extension from URL."""
    parsed = urlparse(url)
    path = parsed.path
    ext = os.path.splitext(path)[1]
    return ext if ext else default


def download_image(url, output_path, session=None):
    """Download an image from URL to output_path."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create session if not provided (for thread safety)
        if session is None:
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        return (url, True, None)
    except Exception as e:
        if output_path.exists():
            output_path.unlink()  # Remove partial download
        return (url, False, str(e))


def process_jsonl(input_file, output_file, data_dir, force_redownload=False, num_workers=8):
    """
    Process JSONL file and download images.
    
    Args:
        input_file: Path to input .jsonl file
        output_file: Path to output .jsonl file with paths
        data_dir: Directory to store downloaded images
        force_redownload: If True, redownload even if file exists
        num_workers: Number of worker threads for downloading (default: 8)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    data_path = Path(data_dir)
    
    # Create subdirectories for different image types
    tgt_dir = data_path / 'tgt'
    src_dir = data_path / 'src'
    inpainted_dir = data_path / 'inpainted'
    
    for d in [tgt_dir, src_dir, inpainted_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # URL to path mapping to handle duplicates
    url_to_path = {}
    
    # First pass: collect all unique URLs and check existing files
    print("Scanning input file and checking for existing downloads...")
    lines = []
    urls_to_download = set()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            lines.append(data)
            
            for url_field in ['tgt_img_url', 'src_img_url', 'inpainted_img_url']:
                url = data.get(url_field)
                if url and url not in url_to_path:
                    # Determine the appropriate directory and filename
                    if url_field == 'tgt_img_url':
                        img_dir = tgt_dir
                    elif url_field == 'src_img_url':
                        img_dir = src_dir
                    else:  # inpainted_img_url
                        img_dir = inpainted_dir
                    
                    # Generate filename from URL hash + extension
                    url_hash = get_url_hash(url)
                    ext = get_file_extension(url)
                    filename = f"{url_hash}{ext}"
                    file_path = img_dir / filename
                    
                    url_to_path[url] = file_path
                    
                    # Check if file already exists
                    if not file_path.exists() or force_redownload:
                        urls_to_download.add(url)
    
    print(f"Total lines: {len(lines)}")
    print(f"Unique URLs: {len(url_to_path)}")
    print(f"Already downloaded: {len(url_to_path) - len(urls_to_download)}")
    print(f"To download: {len(urls_to_download)}")
    
    # Download images with multi-threading
    if urls_to_download:
        print(f"\nDownloading images using {num_workers} worker threads...")
        
        failed_urls = []
        failed_errors = {}
        
        # Create a list of download tasks (url, path)
        download_tasks = [(url, url_to_path[url]) for url in urls_to_download]
        
        # Use ThreadPoolExecutor for concurrent downloads
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all download tasks
            future_to_url = {
                executor.submit(download_image, url, path): url
                for url, path in download_tasks
            }
            
            # Process completed downloads with progress bar
            with tqdm(total=len(download_tasks), desc="Downloading", unit="img") as pbar:
                for future in as_completed(future_to_url):
                    url, success, error = future.result()
                    if not success:
                        failed_urls.append(url)
                        failed_errors[url] = error
                    pbar.update(1)
        
        if failed_urls:
            print(f"\n⚠ Warning: {len(failed_urls)} images failed to download")
            print("Failed URLs will be skipped in the output file.")
            # Optionally show first few errors for debugging
            if len(failed_urls) <= 5:
                for url in failed_urls:
                    print(f"  - {url}: {failed_errors.get(url, 'Unknown error')}")
    
    # Second pass: update lines with paths and write output
    print("\nUpdating JSONL with image paths...")
    successful_lines = 0
    skipped_lines = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for data in tqdm(lines, desc="Processing lines"):
            # Check if all required images exist
            all_exist = True
            for url_field in ['tgt_img_url', 'src_img_url', 'inpainted_img_url']:
                url = data.get(url_field)
                if url:
                    file_path = url_to_path[url]
                    if not file_path.exists():
                        all_exist = False
                        break
            
            if not all_exist:
                skipped_lines += 1
                continue
            
            # Add path fields
            if 'tgt_img_url' in data:
                data['tgt_img_path'] = str(url_to_path[data['tgt_img_url']].replace(data_path, ''))
            if 'src_img_url' in data:
                data['src_img_path'] = str(url_to_path[data['src_img_url']].replace(data_path, ''))
            if 'inpainted_img_url' in data:
                data['inpainted_img_path'] = str(url_to_path[data['inpainted_img_url']].replace(data_path, ''))
            
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            successful_lines += 1
    
    print(f"\n✓ Processing complete!")
    print(f"  Successful lines: {successful_lines}")
    if skipped_lines > 0:
        print(f"  Skipped lines (missing images): {skipped_lines}")
    print(f"  Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download images from .jsonl file with resume and duplicate handling'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/trans_data/annotations.jsonl',
        help='Input .jsonl file (default: annotations.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/trans_data/annotations_with_paths.jsonl',
        help='Output .jsonl file (default: annotations_with_paths.jsonl)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/trans_data/data',
        help='Directory to store downloaded images (default: data)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of worker threads for downloading (default: 8)'
    )
    parser.add_argument(
        '--force-redownload',
        action='store_true',
        help='Force redownload even if files exist'
    )
    
    args = parser.parse_args()
    
    process_jsonl(
        input_file=args.input,
        output_file=args.output,
        data_dir=args.data_dir,
        force_redownload=args.force_redownload,
        num_workers=args.workers
    )


if __name__ == '__main__':
    main()

