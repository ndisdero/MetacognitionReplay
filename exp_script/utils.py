from psychopy import visual, core, event, gui, prefs # mental replay condition counterbalanced, with visuals
import random
def show_images(win, images, min_display_time): # function to display instructions
    for img in images:
        stim = visual.ImageStim(win, image=img)
        timer = core.Clock()
        timer.reset()
        while True:
            stim.draw()
            win.flip()
            keys = event.getKeys(keyList=['space', 'escape'])
            if timer.getTime() > min_display_time:
                if 'space' in keys:
                    break
                if 'escape' in keys:
                    win.close()
                    core.quit()
            else:
                if 'escape' in keys:
                    win.close()
                    core.quit()

# Create visual scales for both conditions
def draw_visual_scale(win, selected, labels, y_offset=-100): # squares offset from labels
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

# Function to generate a pseudorandom motion direction
def get_pseudorandom_direction(prev_directions, max_repeats=3): # a direction can only be repeated as much as "max_repeats"
    if len(prev_directions) < max_repeats:
        return random.choice([-1, 1])
    last_n = prev_directions[-max_repeats:]
    if all(d == last_n[0] for d in last_n):
        return 1 if last_n[0] == -1 else -1
    return random.choice([-1, 1])