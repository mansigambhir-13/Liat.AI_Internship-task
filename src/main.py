# main.py
import os
import json
import argparse
import cv2
from reid_system import PlayerReIdentificationSystem

def main():
    parser = argparse.ArgumentParser(description='Player Re-identification System')
    parser.add_argument('--video', type=str, required=True, 
                       help='Path to input video file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLOv11 model file')
    parser.add_argument('--output_video', type=str, default=None,
                       help='Path to save annotated output video')
    parser.add_argument('--output_json', type=str, default='tracking_results.json',
                       help='Path to save tracking results as JSON')
    parser.add_argument('--similarity_threshold', type=float, default=0.3,
                       help='Similarity threshold for track association')
    parser.add_argument('--reid_threshold', type=float, default=0.4,
                       help='Similarity threshold for re-identification')
    parser.add_argument('--max_inactive_frames', type=int, default=30,
                       help='Maximum frames a track can be inactive before removal')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Create output directory if needed
    if args.output_video:
        output_dir = os.path.dirname(args.output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Create outputs directory for JSON
    json_output_dir = os.path.dirname(args.output_json)
    if json_output_dir and not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)
    
    # Initialize re-identification system
    print("Initializing Player Re-identification System...")
    reid_system = PlayerReIdentificationSystem(
        model_path=args.model,
        similarity_threshold=args.similarity_threshold,
        max_frames_inactive=args.max_inactive_frames
    )
    reid_system.reid_similarity_threshold = args.reid_threshold
    
    # Process video
    print(f"Processing video: {args.video}")
    tracking_results = reid_system.process_video(args.video, args.output_video)
    
    # Save tracking results
    print(f"Saving results to: {args.output_json}")
    save_results(tracking_results, args.output_json)
    
    # Print statistics
    stats = reid_system.get_track_statistics()
    print_statistics(stats)
    
    print("Processing complete!")

def save_results(tracking_results, output_path):
    """Save tracking results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for frame_idx, results in tracking_results.items():
        json_results[str(frame_idx)] = []
        for result in results:
            json_result = {
                'track_id': int(result['track_id']),
                'bbox': [int(x) for x in result['bbox']],
                'confidence': float(result['confidence']),
                'frames_tracked': int(result['frames_tracked']),
                'active': bool(result['active'])
            }
            json_results[str(frame_idx)].append(json_result)
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

def print_statistics(stats):
    """Print tracking statistics"""
    print("\n" + "="*50)
    print("TRACKING STATISTICS")
    print("="*50)
    print(f"Total tracks created: {stats['total_tracks_created']}")
    print(f"Currently active tracks: {stats['currently_active_tracks']}")
    print(f"Inactive tracks: {stats['inactive_tracks']}")
    print(f"Frames processed: {stats['frames_processed']}")
    
    print("\nPER-TRACK DETAILS:")
    print("-" * 30)
    for track in stats['track_details']:
        status = "ACTIVE" if track['active'] else "INACTIVE"
        duration = track['last_seen_frame'] - track['first_seen_frame'] + 1
        print(f"Track {track['track_id']:2d}: {status:8s} | "
              f"Duration: {duration:3d} frames | "
              f"Tracked: {track['total_frames_tracked']:3d} frames | "
              f"Frames {track['first_seen_frame']:3d}-{track['last_seen_frame']:3d}")

if __name__ == "__main__":
    main()
