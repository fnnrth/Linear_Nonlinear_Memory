import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ============================================================================
# Task Definitions
# ============================================================================

class CognitiveTask:
    """Base class for cognitive tasks"""
    BASE_INPUT_DIM = 9  # Maximum input channels needed by any task
    
    def __init__(self, duration_params: Dict):
        self.duration_params = duration_params
    
    def generate_trial(self):
        raise NotImplementedError
    
    def __call__(self):
        """Wrapper that automatically pads inputs to BASE_INPUT_DIM"""
        inputs, targets, mask = self.generate_trial()
        
        # Pad inputs if necessary
        if inputs.shape[1] < self.BASE_INPUT_DIM:
            padding = torch.zeros(inputs.shape[0], self.BASE_INPUT_DIM - inputs.shape[1])
            inputs = torch.cat([inputs, padding], dim=1)
        
        return inputs, targets, mask

class CopyTask(CognitiveTask):
    """Simple copy task: remember a random input pattern and reproduce it after delay"""
    def __init__(self, duration_params: Dict, pattern_dim=3, pattern_length=5):
        super().__init__(duration_params)
        self.pattern_dim = pattern_dim  # Number of input channels used for the pattern
        self.pattern_length = pattern_length  # Number of time steps the pattern is presented
        
    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params['context'])
        T_stim = np.random.randint(*self.duration_params['stimulus'])
        T_delay = np.random.randint(*self.duration_params['delay'])
        T_response = np.random.randint(*self.duration_params['response'])
        T_total = T_context + T_stim + T_delay + T_response

        if T_stim < self.pattern_length:
            raise ValueError(f"Stimulus duration must be at least {self.pattern_length} to present the full pattern.")
        
        # Sample random pattern
        pattern = torch.rand(self.pattern_length, self.pattern_dim)
        num_repeats = (T_stim // self.pattern_length) + 1
        repeated_pattern = pattern.repeat(num_repeats, 1)

        # Create inputs: [fixation, pattern..., rule_copy]
        inputs = torch.zeros(T_total, self.pattern_dim + 2)
        
        # Context period
        inputs[:T_context, 0] = 1  # fixation
        inputs[:T_context, -1] = 1  # rule_copy
        
        # Stimulus period: fixation + pattern + rule
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1:1+self.pattern_dim] = repeated_pattern[:T_stim]
        inputs[t_stim_start:t_stim_end, -1] = 1
        
        # Delay period: fixation + rule
        t_delay_start = t_stim_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1
        inputs[t_delay_start:t_delay_end, -1] = 1
        
        # Response period: no fixation, rule still on
        t_resp_start = t_delay_end
        inputs[t_resp_start:, -1] = 1
        
        # Target output: reproduce the pattern during response period
        targets = torch.zeros(T_total, self.pattern_dim)

        # Mask: only evaluate during response period
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1
        
        return inputs, targets, mask

class DelayedResponse(CognitiveTask):
    """
    Delayed Response task: remember stimulus direction and respond after delay
    Pro: respond in same direction as stimulus
    Anti: respond in opposite direction
    """
    def __init__(self, duration_params: Dict, mode='pro'):
        super().__init__(duration_params)
        self.mode = mode  # 'pro' or 'anti'
        
    def generate_trial(self):
        # Sample durations
        T_context = np.random.randint(*self.duration_params['context'])
        T_stim = np.random.randint(*self.duration_params['stimulus'])
        T_delay = np.random.randint(*self.duration_params['delay'])
        T_response = np.random.randint(*self.duration_params['response'])
        T_total = T_context + T_stim + T_delay + T_response
        
        # Sample stimulus angle
        theta = np.random.uniform(0, 2*np.pi)
        
        # Create inputs: [fixation, stim_cos, stim_sin, rule_pro, rule_anti]
        inputs = torch.zeros(T_total, 5)
        
        # Context period: fixation + rule
        inputs[:T_context, 0] = 1  # fixation
        if self.mode == 'pro':
            inputs[:T_context, 3] = 1  # rule_pro
        else:
            inputs[:T_context, 4] = 1  # rule_anti
            
        # Stimulus period: fixation + stimulus + rule
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1  # fixation
        inputs[t_stim_start:t_stim_end, 1] = np.cos(theta)  # stim_cos
        inputs[t_stim_start:t_stim_end, 2] = np.sin(theta)  # stim_sin
        if self.mode == 'pro':
            inputs[t_stim_start:t_stim_end, 3] = 1
        else:
            inputs[t_stim_start:t_stim_end, 4] = 1
            
        # Delay period: fixation + rule
        t_delay_start = t_stim_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1
        if self.mode == 'pro':
            inputs[t_delay_start:t_delay_end, 3] = 1
        else:
            inputs[t_delay_start:t_delay_end, 4] = 1
            
        # Response period: no fixation, rule still on
        t_resp_start = t_delay_end
        if self.mode == 'pro':
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1
        
        # Target output: [fixation, response_cos, response_sin]
        targets = torch.zeros(3)
        targets[0] = 0  # no fixation during response
        if self.mode == 'pro':
            targets[1] = np.cos(theta)
            targets[2] = np.sin(theta)
        else:  # anti
            targets[1] = np.cos(theta + np.pi)
            targets[2] = np.sin(theta + np.pi)
        
        # Mask: only evaluate during response period
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1
        
        return inputs, targets, mask

class ReactionTime(CognitiveTask):
    """
    Reaction time task: respond immediately to stimulus
    Pro: respond in same direction
    Anti: respond in opposite direction
    """
    def __init__(self, duration_params: Dict, mode='pro'):
        super().__init__(duration_params)
        self.mode = mode
        
    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params['context'])
        T_stim_response = np.random.randint(*self.duration_params['stimulus'])
        T_total = T_context + T_stim_response
        
        theta = np.random.uniform(0, 2*np.pi)
        
        inputs = torch.zeros(T_total, 5)
        
        # Context period
        inputs[:T_context, 0] = 1  # fixation
        if self.mode == 'pro':
            inputs[:T_context, 3] = 1
        else:
            inputs[:T_context, 4] = 1
            
        # Stimulus/Response period: no fixation, stimulus present
        t_resp_start = T_context
        inputs[t_resp_start:, 1] = np.cos(theta)
        inputs[t_resp_start:, 2] = np.sin(theta)
        if self.mode == 'pro':
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1
        
        targets = torch.zeros(3)
        targets[0] = 0
        if self.mode == 'pro':
            targets[1] = np.cos(theta)
            targets[2] = np.sin(theta)
        else:
            targets[1] = np.cos(theta + np.pi)
            targets[2] = np.sin(theta + np.pi)
        
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1
        
        return inputs, targets, mask

class CategoryDecision(CognitiveTask):
    """
    Category decision task: respond based on whether stimulus is above/below threshold
    Pro: respond in category 1 if theta < pi, category 2 if theta > pi
    Anti: opposite
    """
    def __init__(self, duration_params: Dict, mode='pro'):
        super().__init__(duration_params)
        self.mode = mode
        
    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params['context'])
        T_stim = np.random.randint(*self.duration_params['stimulus'])
        T_response = np.random.randint(*self.duration_params['response'])
        T_total = T_context + T_stim + T_response
        
        theta = np.random.uniform(0, 2*np.pi)
        
        inputs = torch.zeros(T_total, 5)
        
        # Context
        inputs[:T_context, 0] = 1
        if self.mode == 'pro':
            inputs[:T_context, 3] = 1
        else:
            inputs[:T_context, 4] = 1
            
        # Stimulus
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = np.cos(theta)
        inputs[t_stim_start:t_stim_end, 2] = np.sin(theta)
        if self.mode == 'pro':
            inputs[t_stim_start:t_stim_end, 3] = 1
        else:
            inputs[t_stim_start:t_stim_end, 4] = 1
            
        # Response
        t_resp_start = t_stim_end
        if self.mode == 'pro':
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1
        
        # Determine category
        if self.mode == 'pro':
            category_angle = 0 if theta < np.pi else np.pi
        else:  # anti
            category_angle = np.pi if theta < np.pi else 0
        
        targets = torch.zeros(3)
        targets[0] = 0
        targets[1] = np.cos(category_angle)
        targets[2] = np.sin(category_angle)
        
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1
        
        return inputs, targets, mask



class DelayedMatchToSample(CognitiveTask):
    """
    Delayed Match-to-Sample task: remember first stimulus, compare to second stimulus
    Match: respond if both stimuli are in same direction (within tolerance)
    NonMatch: respond if stimuli are in different directions
    """
    def __init__(self, duration_params: Dict, mode='match'):
        super().__init__(duration_params)
        self.mode = mode  # 'match' or 'nonmatch'
        self.match_threshold = np.pi / 4  # 45 degrees tolerance
        
    def generate_trial(self):
        # Sample durations
        T_context = np.random.randint(*self.duration_params['context'])
        T_stim1 = np.random.randint(*self.duration_params['stimulus'])
        T_delay1 = np.random.randint(*self.duration_params['delay'])
        T_stim2 = np.random.randint(*self.duration_params['stimulus'])
        T_delay2 = np.random.randint(5, 15)  # Short delay before response
        T_response = np.random.randint(*self.duration_params['response'])
        T_total = T_context + T_stim1 + T_delay1 + T_stim2 + T_delay2 + T_response
        
        # Sample two stimulus angles
        theta1 = np.random.uniform(0, 2*np.pi)
        # Second stimulus: sometimes match, sometimes don't
        if np.random.rand() < 0.5:
            # Match trial
            theta2 = theta1 + np.random.uniform(-self.match_threshold/2, self.match_threshold/2)
            is_match = True
        else:
            # Non-match trial (ensure they're different)
            theta2 = theta1 + np.random.uniform(self.match_threshold, 2*np.pi - self.match_threshold)
            is_match = False
        
        # Wrap angles
        theta2 = theta2 % (2*np.pi)
        
        # Create inputs: [fixation, stim_cos, stim_sin, rule_match, rule_nonmatch]
        inputs = torch.zeros(T_total, 5)
        
        # Context period
        inputs[:T_context, 0] = 1  # fixation
        if self.mode == 'match':
            inputs[:T_context, 3] = 1  # rule_match
        else:
            inputs[:T_context, 4] = 1  # rule_nonmatch
            
        # First stimulus
        t_stim1_start = T_context
        t_stim1_end = t_stim1_start + T_stim1
        inputs[t_stim1_start:t_stim1_end, 0] = 1
        inputs[t_stim1_start:t_stim1_end, 1] = np.cos(theta1)
        inputs[t_stim1_start:t_stim1_end, 2] = np.sin(theta1)
        if self.mode == 'match':
            inputs[t_stim1_start:t_stim1_end, 3] = 1
        else:
            inputs[t_stim1_start:t_stim1_end, 4] = 1
            
        # First delay
        t_delay1_start = t_stim1_end
        t_delay1_end = t_delay1_start + T_delay1
        inputs[t_delay1_start:t_delay1_end, 0] = 1
        if self.mode == 'match':
            inputs[t_delay1_start:t_delay1_end, 3] = 1
        else:
            inputs[t_delay1_start:t_delay1_end, 4] = 1
            
        # Second stimulus
        t_stim2_start = t_delay1_end
        t_stim2_end = t_stim2_start + T_stim2
        inputs[t_stim2_start:t_stim2_end, 0] = 1
        inputs[t_stim2_start:t_stim2_end, 1] = np.cos(theta2)
        inputs[t_stim2_start:t_stim2_end, 2] = np.sin(theta2)
        if self.mode == 'match':
            inputs[t_stim2_start:t_stim2_end, 3] = 1
        else:
            inputs[t_stim2_start:t_stim2_end, 4] = 1
            
        # Second delay
        t_delay2_start = t_stim2_end
        t_delay2_end = t_delay2_start + T_delay2
        inputs[t_delay2_start:t_delay2_end, 0] = 1
        if self.mode == 'match':
            inputs[t_delay2_start:t_delay2_end, 3] = 1
        else:
            inputs[t_delay2_start:t_delay2_end, 4] = 1
            
        # Response period: no fixation
        t_resp_start = t_delay2_end
        if self.mode == 'match':
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1
        
        # Target output: [fixation, response_cos, response_sin]
        # Respond in fixed direction (e.g., 0°) if match, opposite (180°) if non-match
        targets = torch.zeros(3)
        targets[0] = 0  # no fixation
        
        if self.mode == 'match':
            # Match task: respond 0° if match, 180° if non-match
            if is_match:
                targets[1] = 1.0  # cos(0)
                targets[2] = 0.0  # sin(0)
            else:
                targets[1] = -1.0  # cos(π)
                targets[2] = 0.0   # sin(π)
        else:
            # NonMatch task: respond 0° if non-match, 180° if match
            if is_match:
                targets[1] = -1.0  # cos(π)
                targets[2] = 0.0   # sin(π)
            else:
                targets[1] = 1.0  # cos(0)
                targets[2] = 0.0  # sin(0)
        
        # Mask: only evaluate during response period
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1
        
        return inputs, targets, mask

class ContextIntegration(CognitiveTask):
    """
    Context-dependent integration: Two stimuli presented sequentially,
    integrate only the relevant modality based on context
    
    - ContextMod1: integrate modality 1, ignore modality 2
    - ContextMod2: integrate modality 2, ignore modality 1
    
    This tests: selective attention + integration of noisy evidence
    """
    def __init__(self, duration_params: Dict, relevant_modality=1):
        super().__init__(duration_params)
        self.relevant_modality = relevant_modality  # 1 or 2
        
    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params['context'])
        T_stim1 = np.random.randint(*self.duration_params['stimulus'])
        T_stim2 = np.random.randint(*self.duration_params['stimulus'])
        T_delay = np.random.randint(10, 20)  # Short delay
        T_response = np.random.randint(*self.duration_params['response'])
        T_total = T_context + T_stim1 + T_stim2 + T_delay + T_response
        
        # Sample angles and coherences for both modalities
        # Relevant modality has stronger signal
        theta_relevant = np.random.uniform(0, 2*np.pi)
        coherence_relevant = 0.8
        
        # Irrelevant modality has different angle, weaker signal
        theta_irrelevant = np.random.uniform(0, 2*np.pi)
        coherence_irrelevant = 0.3
        
        # Create inputs: [fixation, mod1_cos, mod1_sin, mod2_cos, mod2_sin, rule_mod1, rule_mod2]
        inputs = torch.zeros(T_total, 7)
        
        # Context period
        inputs[:T_context, 0] = 1  # fixation
        if self.relevant_modality == 1:
            inputs[:T_context, 5] = 1  # rule_mod1
        else:
            inputs[:T_context, 6] = 1  # rule_mod2
        
        # First stimulus presentation
        t_stim1_start = T_context
        t_stim1_end = t_stim1_start + T_stim1
        inputs[t_stim1_start:t_stim1_end, 0] = 1
        
        # Modality 1
        inputs[t_stim1_start:t_stim1_end, 1] = coherence_relevant * np.cos(theta_relevant) if self.relevant_modality == 1 else coherence_irrelevant * np.cos(theta_irrelevant)
        inputs[t_stim1_start:t_stim1_end, 2] = coherence_relevant * np.sin(theta_relevant) if self.relevant_modality == 1 else coherence_irrelevant * np.sin(theta_irrelevant)
        
        # Modality 2
        inputs[t_stim1_start:t_stim1_end, 3] = coherence_irrelevant * np.cos(theta_irrelevant) if self.relevant_modality == 1 else coherence_relevant * np.cos(theta_relevant)
        inputs[t_stim1_start:t_stim1_end, 4] = coherence_irrelevant * np.sin(theta_irrelevant) if self.relevant_modality == 1 else coherence_relevant * np.sin(theta_relevant)
        
        if self.relevant_modality == 1:
            inputs[t_stim1_start:t_stim1_end, 5] = 1
        else:
            inputs[t_stim1_start:t_stim1_end, 6] = 1
        
        # Second stimulus presentation (adds more evidence)
        t_stim2_start = t_stim1_end
        t_stim2_end = t_stim2_start + T_stim2
        inputs[t_stim2_start:t_stim2_end, 0] = 1
        
        # Same relevant/irrelevant structure
        inputs[t_stim2_start:t_stim2_end, 1] = coherence_relevant * np.cos(theta_relevant) if self.relevant_modality == 1 else coherence_irrelevant * np.cos(theta_irrelevant)
        inputs[t_stim2_start:t_stim2_end, 2] = coherence_relevant * np.sin(theta_relevant) if self.relevant_modality == 1 else coherence_irrelevant * np.sin(theta_irrelevant)
        inputs[t_stim2_start:t_stim2_end, 3] = coherence_irrelevant * np.cos(theta_irrelevant) if self.relevant_modality == 1 else coherence_relevant * np.cos(theta_relevant)
        inputs[t_stim2_start:t_stim2_end, 4] = coherence_irrelevant * np.sin(theta_irrelevant) if self.relevant_modality == 1 else coherence_relevant * np.sin(theta_relevant)
        
        if self.relevant_modality == 1:
            inputs[t_stim2_start:t_stim2_end, 5] = 1
        else:
            inputs[t_stim2_start:t_stim2_end, 6] = 1
        
        # Delay
        t_delay_start = t_stim2_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1
        if self.relevant_modality == 1:
            inputs[t_delay_start:t_delay_end, 5] = 1
        else:
            inputs[t_delay_start:t_delay_end, 6] = 1
        
        # Response period
        t_resp_start = t_delay_end
        if self.relevant_modality == 1:
            inputs[t_resp_start:, 5] = 1
        else:
            inputs[t_resp_start:, 6] = 1
        
        # Target: respond in direction of relevant modality
        targets = torch.zeros(3)
        targets[0] = 0
        targets[1] = np.cos(theta_relevant)
        targets[2] = np.sin(theta_relevant)
        
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1
        
        return inputs, targets, mask


class GoNogo(CognitiveTask):
    """
    Go/Nogo task: Simple decision task
    - Go: respond in a fixed direction when stimulus amplitude > threshold
    - Nogo: maintain fixation when stimulus amplitude < threshold
    
    This tests: simple threshold decision + motor control
    """
    def __init__(self, duration_params: Dict):
        super().__init__(duration_params)
        self.threshold = 0.5  # Amplitude threshold
        
    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params['context'])
        T_stim = np.random.randint(*self.duration_params['stimulus'])
        T_response = np.random.randint(*self.duration_params['response'])
        T_total = T_context + T_stim + T_response
        
        # Random angle and amplitude
        theta = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(0.2, 1.0)
        is_go = amplitude > self.threshold
        
        # Create inputs: [fixation, stim_cos, stim_sin, rule_gonogo]
        inputs = torch.zeros(T_total, 4)
        
        # Context period
        inputs[:T_context, 0] = 1
        inputs[:T_context, 3] = 1  # rule
        
        # Stimulus period
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = amplitude * np.cos(theta)
        inputs[t_stim_start:t_stim_end, 2] = amplitude * np.sin(theta)
        inputs[t_stim_start:t_stim_end, 3] = 1
        
        # Response period
        t_resp_start = t_stim_end
        inputs[t_resp_start:, 3] = 1
        
        # Target: if go, respond in fixed direction (0°); if nogo, maintain fixation
        targets = torch.zeros(3)
        if is_go:
            targets[0] = 0  # break fixation
            targets[1] = 1.0  # cos(0)
            targets[2] = 0.0  # sin(0)
        else:
            targets[0] = 1  # maintain fixation
            targets[1] = 0.0
            targets[2] = 0.0
        
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1
        
        return inputs, targets, mask
    


# ============================================================================
# Dataset
# ============================================================================

class MultiTaskDataset(Dataset):
    """Dataset that generates trials from multiple tasks"""
    def __init__(self, tasks: List[CognitiveTask], n_trials: int = 1000):
        self.tasks = tasks
        self.n_trials = n_trials
        
    def __len__(self):
        return self.n_trials
    
    def __getitem__(self, idx):
    # Randomly select a task
        task_idx = np.random.randint(len(self.tasks))
        task = self.tasks[task_idx]
        
        # Generate trial (automatically padded via __call__)
        inputs, targets, mask = task()  # Use __call__ instead of generate_trial()
        
        # Add task identity to inputs (one-hot encoding)
        task_id = torch.zeros(inputs.shape[0], len(self.tasks))
        task_id[:, task_idx] = 1
        inputs = torch.cat([inputs, task_id], dim=1)
        
        return inputs, targets, mask, task_idx


class ContinuousMultitaskDataset(MultiTaskDataset):
    """
    A wrapper around MultiTaskDataset that allows training on ONLY ONE task at a time,
    while keeping the input dimensions correct for the GLOBAL set of tasks.
    """
    def __init__(self, tasks, n_trials=1000):
        super().__init__(tasks, n_trials)
        self.active_task_idx = 0  # Default to the first task
        
    def set_active_task(self, task_idx):
        """Switch the dataset to generate trials only for this specific task index."""
        if 0 <= task_idx < len(self.tasks):
            self.active_task_idx = task_idx
        else:
            raise ValueError(f"Invalid task index {task_idx}. Max is {len(self.tasks)-1}")

    def __getitem__(self, idx):
        # 1. Always select the currently ACTIVE task
        task = self.tasks[self.active_task_idx]
        
        # 2. Generate trial (inputs has shape [T, 7])
        inputs, targets, mask = task() 
        
        # 3. Add Task ID: CRITICAL STEP
        # We create a one-hot vector with the size of ALL tasks (Global dimension),
        # even though we are only training on one specific task right now.
        task_id = torch.zeros(inputs.shape[0], len(self.tasks))
        task_id[:, self.active_task_idx] = 1
        
        # 4. Concatenate
        inputs = torch.cat([inputs, task_id], dim=1)
        
        return inputs, targets, mask, self.active_task_idx

def collate_fn(batch):
    """Custom collate function to handle variable length sequences and input dims"""
    inputs_list, targets_list, masks_list, task_ids = zip(*batch)
    
    # Find max length and max input dimension
    max_len = max(inp.shape[0] for inp in inputs_list)
    max_input_dim = max(inp.shape[1] for inp in inputs_list)
    batch_size = len(batch)
    
    # Pad sequences
    inputs_padded = torch.zeros(batch_size, max_len, max_input_dim)
    masks_padded = torch.zeros(batch_size, max_len)
    
    for i, (inp, mask) in enumerate(zip(inputs_list, masks_list)):
        T = inp.shape[0]
        input_dim = inp.shape[1]
        inputs_padded[i, :T, :input_dim] = inp
        masks_padded[i, :T] = mask
    
    targets = torch.stack(targets_list)
    task_ids = torch.tensor(task_ids)
    
    return inputs_padded, targets, masks_padded, task_ids

'''
def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    inputs_list, targets_list, masks_list, task_ids = zip(*batch)
    
    # Find max length
    max_len = max(inp.shape[0] for inp in inputs_list)
    batch_size = len(batch)
    input_dim = inputs_list[0].shape[1]
    
    # Pad sequences
    inputs_padded = torch.zeros(batch_size, max_len, input_dim)
    masks_padded = torch.zeros(batch_size, max_len)
    
    for i, (inp, mask) in enumerate(zip(inputs_list, masks_list)):
        T = inp.shape[0]
        inputs_padded[i, :T] = inp
        masks_padded[i, :T] = mask
    
    targets = torch.stack(targets_list)
    task_ids = torch.tensor(task_ids)
    
    return inputs_padded, targets, masks_padded, task_ids
'''