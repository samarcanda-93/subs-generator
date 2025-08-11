"""Progress tracking and visualization."""

import time
import threading
from tqdm import tqdm


class ProgressTracker:
    """Manages progress tracking for concurrent video processing."""
    
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.completed_files = {"count": 0}
        self.progress_bar = None
        self.heartbeat_thread = None
        self.paused = False
        
    def start(self):
        """Start progress tracking with animated heartbeat."""
        # Use a simple counter format without percentage bar
        self.progress_bar = tqdm(
            total=self.total_files, 
            desc="Processing videos", 
            unit="file",
            bar_format="{desc}: {n_fmt}/{total_fmt} files | {elapsed} | {postfix}",
            ncols=80
        )
        
        # Start heartbeat animation
        self.heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.heartbeat_thread.start()
    
    def update(self, increment: int = 1):
        """Update progress bar when a file completes."""
        self.completed_files["count"] += increment
        if self.progress_bar:
            self.progress_bar.update(increment)
            if self.completed_files["count"] == self.total_files:
                self.progress_bar.set_postfix_str("✓ Complete")
    
    def pause(self):
        """Pause the progress spinner for user input."""
        self.paused = True
        if self.progress_bar:
            # Clear the current line and move cursor to beginning
            print("\r" + " " * 80 + "\r", end="", flush=True)
    
    def resume(self):
        """Resume the progress spinner after user input."""
        self.paused = False
    
    def stop(self):
        """Stop progress tracking and cleanup."""
        if self.progress_bar:
            self.completed_files["count"] = self.total_files  # Stop heartbeat
            self.progress_bar.set_postfix_str("✓ Complete")
            self.progress_bar.close()
    
    def _heartbeat(self):
        """Animated activity indicator while processing."""
        spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0
        
        while self.completed_files["count"] < self.total_files:
            # Skip animation if paused
            if not self.paused:
                spinner = spinners[spinner_idx % len(spinners)]
                if self.progress_bar:
                    self.progress_bar.set_postfix_str(f"{spinner} Processing...")
            spinner_idx += 1
            time.sleep(0.1)  # Faster animation