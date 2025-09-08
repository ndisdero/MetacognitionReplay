from psychopy import visual, core, event, gui, prefs # mental replay condition counterbalanced, with visuals

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