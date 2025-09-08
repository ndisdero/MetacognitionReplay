from psychopy import visual, core, event, gui, prefs # mental replay condition counterbalanced, with visuals
import random, os, csv
from datetime import datetime
import utils
import gabor_patches.generate_gaborPatches as gabor
import numpy as np


# Window settings
prefs.general['winType'] = 'pyglet'
win = visual.Window([800, 600], color='gray', units='pix', fullscr=False)
key_list = ["v", "b"]

# Experiment parameters
N_TRAINING_TRIALS = 9  # number of training trials
TOTAL_BLOCKS = 4  # must be even: half with replay, half without
NUMBER_OF_TRIALS = 2  # per block
STIMULUS_DURATION = 0.25
INTERTRIAL_PAUSE = 2  # fixation cross display time
MENTAL_REPLAY_PAUSE = 3  # pause during mental replay instruction
FEEDBACK_TIME = 1

# Get path to folder where the script is located
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()  # fallback if file is not available

# Set relative paths
save_directory = os.path.join(base_dir, "data")
image_dir = os.path.join(base_dir, "./images")

# Make sure the data folder exists
os.makedirs(save_directory, exist_ok=True)

# Instruction image setup
image_intro = [os.path.join(image_dir, f) for f in ["Intro1.JPG", "Intro2.JPG", "Intro3.JPG"]]
image_task1 = [os.path.join(image_dir, f) for f in ["Task1_1.JPG"]]
image_task2 = [os.path.join(image_dir, f) for f in ["Task2_1.JPG", "Task2_2.JPG"]]
image_end = [os.path.join(image_dir, f) for f in ["End.JPG"]]

min_display_time = 2  # seconds before intructions can be skipped

# TRIAL PHASE
if TOTAL_BLOCKS % 2 != 0:
    print("TOTAL_BLOCKS must be an even number.")
    core.quit()

# Step 1: Get Participant ID
id_dialog = gui.Dlg(title="Enter Participant ID")
id_dialog.addField("Participant ID:")
id_dialog.show()
if not id_dialog.OK:
    core.quit()
participant_id = str(id_dialog.data[0]).strip()
while not participant_id:
    print("Participant ID cannot be empty.")
    participant_id = str(gui.Dlg(title="Enter Participant ID").addField("Participant ID:").show()[0]).strip()
    if not participant_id:
        core.quit()
participant_id_clean = participant_id.replace(":", "_").replace("/", "_").replace("\\", "_").strip()
random_seed = sum(ord(c) for c in participant_id_clean) # generate random seed
random.seed(random_seed)



# Sorting participants into 8 conditions
participant_id_num = int(participant_id_clean)  # participant ID must be numeric
group_id = participant_id_num % 8  # assign participant to 1 of 8 groups (0–7)

# Factor 1: starting condition
start_with_replay = (group_id % 2 == 0)  # even groups: replay first, odd groups: non-replay first

# Factor 2: scale order
vividness_first = (group_id // 2) % 2 == 0  # groups 0–1,4–5 vividness first; groups 2–3,6–7 confidence first

# Factor 3: hand assignment
left_for_confidence = (group_id // 4) % 2 == 0  # groups 0–3: left hand = confidence; groups 4–7: left hand = vividness

half_blocks = TOTAL_BLOCKS // 2
if start_with_replay:
    block_order = ["with_mental_replay"] * half_blocks + ["without_mental_replay"] * half_blocks
else:
    block_order = ["without_mental_replay"] * half_blocks + ["with_mental_replay"] * half_blocks

test_directions = []

print(f"Assigned to group {group_id}: "
      f"{'Replay first' if start_with_replay else 'Non-replay first'}, "
      f"{'Vividness first' if vividness_first else 'Confidence first'}, "
      f"{'Left=confidence' if left_for_confidence else 'Left=vividness'}")


# Step 2: Collect demographic info
participant_dialog = gui.Dlg(title="Participant Info")
participant_dialog.addField("Gender", choices=["Male", "Female", "Other"])
participant_dialog.addField("Age")
participant_dialog.addField("Handedness", choices=["Left", "Right", "Ambidextrous"])
participant_dialog.show()
if not participant_dialog.OK:
    core.quit()
gender = participant_dialog.data[0]
age_raw = participant_dialog.data[1]
handedness = participant_dialog.data[2]
age = int(age_raw) if age_raw.isdigit() else "NA"


# Set up output file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_file = os.path.join(save_directory, f"{participant_id_clean}_{timestamp}_experiment_data.csv")

# Column headers for the CSV file
header = [
    "participant_id", "gender", "age", "handedness",
    "block", "block_number", "trial", "global_trial",
    "response", "reference", "vividness", "confidence", "response_time",
    "gabor_direction", "correct", "coherence", "vividness_on_left"
]

def save_trial_data(filepath, header, data_row): 
    file_exists = os.path.exists(filepath)
    try:
        with open(filepath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(data_row)
    except PermissionError:
        print(f"Unable to write to file {filepath}. Close the file if it's open.")
        core.quit()

# Function to generate a pseudorandom motion direction
def get_pseudorandom_direction(prev_directions, max_repeats=3): # a direction can only be repeated as much as "max_repeats"
    if len(prev_directions) < max_repeats:
        return random.choice([-1, 1])
    last_n = prev_directions[-max_repeats:]
    if all(d == last_n[0] for d in last_n):
        return 1 if last_n[0] == -1 else -1
    return random.choice([-1, 1])

# Define visual stimuli
DotMotion = visual.DotStim(
    win=win, nDots=200, speed=0.05, dir=0, coherence=0.5,
    fieldSize=400, fieldShape='circle', dotSize=4, color='white',
    signalDots='same', noiseDots='direction', dotLife=5
)
fixation_cross = visual.TextStim(win, text="+", color='white', height=30)
dot_instruction = visual.TextStim(
    win, text="Which direction are most of the dots moving?\nPress V or B.",
    color='white', height=20, pos=(0, -250), wrapWidth=700
)
replay_text = visual.TextStim(win, text="Mentally replay the motion you just saw.", color='white', height=20)

# Response key mappings
left_vividness_keys = {'a': 1, 'z': 2, 'e': 3, 'r': 4} # counterbalancing: the keys switch -- half the time vividness is on the left hand
right_vividness_keys = {'u': 1, 'i': 2, 'o': 3, 'p': 4}
left_confidence_keys = {'a': 1, 'z': 2, 'e': 3, 'r': 4}
right_confidence_keys = {'u': 1, 'i': 2, 'o': 3, 'p': 4}
response_keys = ['v', 'b', 'escape']

# Create visual scales for both conditions
def draw_visual_scale(selected, labels, y_offset=-100): # squares offset from labels
    spacing = 150
    start_x = -((len(labels) - 1) * spacing) / 2
    rect_size = 50

    for i, label in enumerate(labels):
        x = start_x + i * spacing
        # Draw square
        color = 'blue' if selected == (i + 1) else None
        square = visual.Rect(win, width=rect_size, height=rect_size, lineColor='white',
                             fillColor=color, pos=(x, y_offset))
        square.draw()

        # Draw label below
        text = visual.TextStim(win, text=label, pos=(x, y_offset - 40),
                               height=16, color='blue', wrapWidth=200)
        text.draw()


# Main trial procedure
def run_trial(block_type, block_number, trial_num, global_trial, gabor_direction, stim_strength,
              ask_vividness=False, ask_confidence=False, give_feedback=False):
    DotMotion.dir = gabor_direction
    response = None
    rt_clock = core.Clock()
    
    # Set rating key mappings
    if block_type == "with_mental_replay":
        vividness_on_left = not left_for_confidence  # vividness left if confidence is right
        if left_for_confidence:
            confidence_keys = left_confidence_keys
            vividness_keys = right_vividness_keys
            confidence_prompt_text = "How confident are you?\n\nUse A/Z/E/R (left hand)"
            vividness_prompt_text = "How vivid was your mental replay?\n\nUse U/I/O/P (right hand)"
        else:
            confidence_keys = right_confidence_keys
            vividness_keys = left_vividness_keys
            confidence_prompt_text = "How confident are you?\n\nUse U/I/O/P (right hand)"
            vividness_prompt_text = "How vivid was your mental replay?\n\nUse A/Z/E/R (left hand)"
    else:  # without mental replay
        vividness_on_left = None
        confidence_keys = right_confidence_keys
        confidence_prompt_text = (
            "How confident are you in your response?\n\n"
            "U = Not confident   I = Slightly   O = Moderately   P = Very confident"
        )
        vividness_keys = None
        vividness_prompt_text = None

    # Store prompts in proper order
    prompts = []
    if block_type == "with_mental_replay":
        if vividness_first:
            prompts = [("vividness", vividness_prompt_text, vividness_keys),
                       ("confidence", confidence_prompt_text, confidence_keys)]
        else:
            prompts = [("confidence", confidence_prompt_text, confidence_keys),
                       ("vividness", vividness_prompt_text, vividness_keys)]
    elif block_type == "without_mental_replay":
        prompts = [("confidence", confidence_prompt_text, confidence_keys)]

    reference = np.random.randint(0, 180)
    to_show = gabor.generate_gabor_patches(win, reference=reference,
                                    direction=gabor_direction, distance_to_bound=stim_strength)
    
    # Display the Gabor patches
    for stim in to_show[0]:
        stim.draw()
    # And the arcs
    for stim in to_show[1]:
        stim.draw()

    win.flip()
    core.wait(STIMULUS_DURATION)
    response_time = rt_clock.getTime()
    correct_response = 'v' if gabor_direction < 0 else 'b' # it is orange response

    event.clearEvents()
    rt_clock.reset()
    for stim in to_show[1]:
        stim.draw()
    win.flip()

    while True: # Wait for participant's response
        key = event.getKeys(keyList=key_list, timeStamped=rt_clock)
        if key != []:
            info_response = key[0] # take the first response
            response = info_response[0]
            response_time = info_response[1]
            break
    correct = correct_response == response
    
    # Feedback
    if give_feedback:
        feedback_text = 'Correct!' if correct else 'Wrong!' if response else 'No response'
        feedback = visual.TextStim(win, text=feedback_text, pos=(0, 0), color='white')
        feedback.draw()
        win.flip()
        core.wait(FEEDBACK_TIME)

    # --- Ratings loop ---
    ratings = {"vividness": "NA", "confidence": "NA"}

    for measure, prompt_text, keymap in prompts:
        if measure == "vividness":
            replay_text.draw()
            win.flip()
            core.wait(MENTAL_REPLAY_PAUSE)

        prompt = visual.TextStim(
            win, text=prompt_text, color='white', height=20, wrapWidth=700
        )
        rating = None
        event.clearEvents()
        while rating is None:
            prompt.draw()
            labels = ["Not vivid", "Slightly", "Moderately", "Very vivid"] if measure == "vividness" \
                     else ["Not confident", "Slightly", "Moderately", "Very confident"]
            draw_visual_scale(None, labels)
            win.flip()
            keys = event.getKeys()
            for key in keys:
                if key in keymap:
                    rating = keymap[key]
                elif key == 'escape':
                    win.close()
                    core.quit()
            if rating:
                prompt.draw()
                draw_visual_scale(rating, labels)
                win.flip()
                core.wait(0.5)
        ratings[measure] = rating

    # Save data
    row = [
        participant_id, gender, age, handedness,
        block_type, block_number, trial_num, global_trial,
        response,
        reference,
        ratings["vividness"],
        ratings["confidence"],
        response_time, gabor_direction, correct, stim_strength,
        vividness_on_left if block_type == "with_mental_replay" else None
    ]
    save_trial_data(data_file, header, row)


# TRAINING PHASE
def training_part():
    utils.show_images(win, image_intro, min_display_time)
    training_directions = []

    for trial in range(N_TRAINING_TRIALS):
        direction = get_pseudorandom_direction(prev_directions=training_directions)
        training_directions.append(direction)
        run_trial(
            block_type="training",
            block_number=0,
            trial_num=trial + 1,
            global_trial=trial + 1,
            gabor_direction=direction,
            coherence=0.5,
            ask_vividness=False,
            ask_confidence=False,
            give_feedback=True
        )

# Adaptive coherence for test phase

for block_idx, condition in enumerate(block_order):
    if condition == "without_mental_replay":
        utils.show_images(win, image_task1, min_display_time)
    else:
        utils.show_images(win, image_task2, min_display_time)

    stim_strength = 20
    for trial in range(NUMBER_OF_TRIALS):
        fixation_cross.draw()
        win.flip()
        core.wait(INTERTRIAL_PAUSE)

        direction = get_pseudorandom_direction(prev_directions=test_directions)
        test_directions.append(direction)

        block_number = block_idx + 1
        global_trial_number = block_idx * NUMBER_OF_TRIALS + trial + 1 # sets up the total number of trials
        run_trial(
            block_type=condition,
            block_number=block_number,
            trial_num=trial + 1,
            global_trial=global_trial_number,
            gabor_direction=direction,
            stim_strength=stim_strength,
            ask_vividness=(condition == "with_mental_replay"),
            ask_confidence=True,
            give_feedback=False
        )

        # Adapt coherence
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
                last_row = lines[-1].strip().split(',')
                last_correct = last_row[13]
        except Exception:
            last_correct = 'None'

        """if last_correct == 'True':
            trial_coherence = max(0.1, trial_coherence - coherence_step)
        elif last_correct == 'False':
            trial_coherence = min(1.0, trial_coherence + coherence_step)
"""
utils.show_images(win, image_end, min_display_time)
win.close()
core.quit()
