import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent # Import WindowEvent from pyboy.utils
import time
from PIL import Image

class TetrisEnv:
    """
    PyBoy Tetris environment wrapper for reinforcement learning
    """
    # Tetris-specific constants
    TETRIS_ROM_PATH = None  # Set this to your Tetris ROM path
    # Score is stored at $C0A0-$C0A2 as 3-byte little-endian BCD
    TETRIS_SCORE_ADDR = 0xC0A0
    SCREEN_WIDTH = 160
    SCREEN_HEIGHT = 144
    
    # Action space mapping to Game Boy buttons
    ACTIONS = {
        0: "NONE",
        1: "RIGHT",
        2: "LEFT",
        3: "UP",
        4: "DOWN",
        5: "A",
        6: "B",
        7: "START",
        8: "SELECT"
    }
    
    def __init__(self, rom_path=None, headless=True, down_sample=2, frameskip=4):
        """
        Initialize the Tetris environment
        
        Args:
            rom_path (str): Path to the Tetris ROM file
            headless (bool): Whether to run in headless mode
            down_sample (int): Factor by which to downsample screen
            frameskip (int): Number of frames to skip between observations
        """
        self.rom_path = rom_path or self.TETRIS_ROM_PATH
        if self.rom_path is None:
            raise ValueError("ROM path must be provided")
            
        self.headless = headless
        self.down_sample = down_sample
        self.frameskip = frameskip  # New parameter for frameskipping
        # Use the new 'window' argument and 'null' for headless
        self.window_setting = "null" if headless else "SDL2"
        self.pyboy = None
        self.game_over = False
        self.prev_score = 0
        self.frame_count = 0
        
    def _get_screen(self):
        """Extract screen data as numpy array and downsample"""
        # Get screen as PIL Image and convert to numpy array
        screen_image = self.pyboy.screen.image
        screen = np.array(screen_image)
        
        # Down sample if requested
        if self.down_sample > 1:
            h, w, c = screen.shape
            new_h, new_w = h // self.down_sample, w // self.down_sample
            # Use PIL for resizing before converting back to numpy
            screen_image_resized = screen_image.resize((new_w, new_h))
            screen = np.array(screen_image_resized)
            
        # Normalize pixel values
        if screen.max() > 1.0: # Check if normalization is needed
            screen = screen / 255.0
        return screen
        
    def _get_score(self):
        """Extract score from game memory using correct BCD encoding
        
        The score is stored at WRAM location $C0A0-$C0A2 as a 3-byte little-endian BCD number
        Example: score 123456 would be stored as:
        $C0A0: $56 (least significant byte)
        $C0A1: $34 (middle byte)
        $C0A2: $12 (most significant byte)
        """
        try:
            # Read the 3 bytes from memory
            byte0 = self.pyboy.memory[self.TETRIS_SCORE_ADDR]     # Least significant (ones and tens)
            byte1 = self.pyboy.memory[self.TETRIS_SCORE_ADDR + 1] # Middle (hundreds and thousands)
            byte2 = self.pyboy.memory[self.TETRIS_SCORE_ADDR + 2] # Most significant (ten thousands and hundred thousands)
            
            # Convert from BCD to decimal
            # In BCD, each nibble (4 bits) represents a decimal digit
            digit1 = byte0 & 0x0F        # ones
            digit2 = (byte0 & 0xF0) >> 4 # tens
            digit3 = byte1 & 0x0F        # hundreds
            digit4 = (byte1 & 0xF0) >> 4 # thousands
            digit5 = byte2 & 0x0F        # ten thousands
            digit6 = (byte2 & 0xF0) >> 4 # hundred thousands
            
            # Combine digits to form the complete score
            # Each digit represents a power of 10
            score = (digit1 * 1) + (digit2 * 10) + (digit3 * 100) + \
                    (digit4 * 1000) + (digit5 * 10000) + (digit6 * 100000)
            
            return score
            
        except (AttributeError, IndexError) as e:
            # Fallback: if memory access fails, return a dummy score based on frames
            print(f"Warning: Could not access memory: {e}")
            return self.frame_count / 100  # Dummy score based on survival time
        
    def _is_game_over(self):
        """Detect game over state"""
        # This will need game-specific detection logic
        # Could be based on specific memory values or screen patterns
        # Placeholder implementation
        return False
        
    def reset(self):
        """Reset the environment and return initial observation"""
        if self.pyboy:
            self.pyboy.stop()
            
        # Start PyBoy with the Tetris ROM using the updated window setting
        self.pyboy = PyBoy(self.rom_path, window=self.window_setting, sound_emulated=False, scale=1)
        self.pyboy.set_emulation_speed(0)  # Run at maximum speed
        # self.pyboy.start() # Removed deprecated start() call
        
        # Navigate through menus to start the game
        # Use the imported WindowEvent
        time.sleep(10)  # Wait for game to load
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        self.pyboy.tick()
        time.sleep(1)  # Wait for menu
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        
        # Run a few frames to stabilize
        for _ in range(10):
            self.pyboy.tick()
            
        self.game_over = False
        self.prev_score = 0
        self.frame_count = 0
        
        # Return initial observation
        return self._get_screen()
        
    def step(self, action):
        """
        Execute action and return new state, reward, done, info
        
        Args:
            action (int): Action index from ACTIONS dict
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.game_over:
            return self._get_screen(), 0, True, {"score": self.prev_score}
            
        # Send action to emulator
        action_name = self.ACTIONS.get(action, "NONE")
        if action_name != "NONE":
            # Use the imported WindowEvent
            press_event = getattr(WindowEvent, f"PRESS_BUTTON_{action_name}", None)
            release_event = getattr(WindowEvent, f"RELEASE_BUTTON_{action_name}", None)
            
            if press_event and release_event:
                self.pyboy.send_input(press_event)
                self.pyboy.tick()
                self.pyboy.send_input(release_event)
        
        # Run a few frames to see the effect of the action
        for _ in range(self.frameskip):
            self.pyboy.tick()
            
        # Get new state
        next_state = self._get_screen()
        
        # Calculate reward
        current_score = self._get_score()
        reward = current_score - self.prev_score
        self.prev_score = current_score
        
        # Add small negative reward per frame to encourage faster completion
        reward -= 0.01
        
        # Check if game is over
        self.game_over = self._is_game_over()
        
        # Increment frame counter
        self.frame_count += 1
        
        # For very long episodes, force termination
        if self.frame_count > 10000:
            self.game_over = True
            
        info = {
            "score": current_score,
            "frame": self.frame_count
        }
        
        return next_state, reward, self.game_over, info
        
    def close(self):
        """Close the environment"""
        if self.pyboy:
            self.pyboy.stop()
            self.pyboy = None