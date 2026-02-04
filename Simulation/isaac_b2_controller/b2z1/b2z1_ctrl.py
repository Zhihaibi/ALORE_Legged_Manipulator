import torch
import carb

base_vel_cmd_input = [0, 0, 0]

# Update sub_keyboard_event to modify specific rows of the tensor based on key inputs
def sub_keyboard_event(event) -> bool:
    global base_vel_cmd_input
    lin_vel = 0.6
    ang_vel = 0.6
    
    if base_vel_cmd_input is not None:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Update tensor values for environment 0
            if event.input.name == 'W':
                base_vel_cmd_input = [lin_vel, 0, 0]
            elif event.input.name == 'S':
                base_vel_cmd_input = [-lin_vel, 0, 0]
            elif event.input.name == 'A':
                base_vel_cmd_input = [0, 0, ang_vel]
            elif event.input.name == 'D':
                base_vel_cmd_input = [0, 0, -ang_vel]
            elif event.input.name == 'Q':
                base_vel_cmd_input = [0, 0, 0]
            
            # If there are multiple environments, handle inputs for env 1
            if base_vel_cmd_input.shape[0] > 1:
                if event.input.name == 'I':
                    base_vel_cmd_input[1] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
                elif event.input.name == 'K':
                    base_vel_cmd_input[1] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
                elif event.input.name == 'J':
                    base_vel_cmd_input[1] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
                elif event.input.name == 'L':
                    base_vel_cmd_input[1] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
                elif event.input.name == 'M':
                    base_vel_cmd_input[1] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
                elif event.input.name == '>':
                    base_vel_cmd_input[1] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
        
        # Reset commands to zero on key release
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            base_vel_cmd_input.zero_()
    return True

