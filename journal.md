# Developer's Journal
This journal is meant to serve as a log for what we did on this project
at various points in time. It is our intention to use this both as a source of 
reference, as well as a useful trove of information when we write our paper
and presentation on the project.  
### Weekend 23-24 Sept 2023 ###
Did the initial set-up. Made a game.py file which is meant to be the game API
for the learner and added some initial functions, such as start/action/restart.
All of these functions assume the game window dimensions are hard-coded and at 
the dead center of the screen. This is not ideal, and I want to fix this as well 
as the fact that the restart function doesn't work; the solution is CV. I am currently 
mirroring my phone to run the actual game via phone link, this is not ideal because:
- Windows Phone Link is slow and extremely laggy
- I have to use my phone to test it, which requires internet (and so ads on the game which
slows us down and adds another thing the game api/learner have to deal with)
I need to fix this as well.

### Weekend 30-01 Sept/Oct 2023###
Worked on implementing CV and auto-detecting where the screen is. I decided to use opencv
and contours with approxPolyDP to find the screen and make game.py work of off this information.
Unfortunately I was having incredible trouble getting approxPolyDP to detect the proper
rectangle that is our screen. The solution in the end was to use the contours, to find a properly
sized-convex hull as the contours (when we move interior/nested contours) perfectly drew the shape
of window. The process was we first gray scaled the image, then made it black-white depending on 
whether the average rgb value is greater than a given threshold (This made the window
clearly visible as a rectangle) and then running the contour finder on it. Once it found
the contours we iterated over all the convex hulls these contours made it and filtered them
to the 4-sided ones that fit our very specific demands (We assume this characteristic is
unique to the subway surf game window), and return its information. This fixed it reliably.  

I did a simple template matching for the restart function to 
identify the play button (which only appears in the restart menu) and added a second delay as
if we click on the play button to early after restart, it will ignore the click. I moved all the
CV stuff to a new file called view.py At the end of this week I have a functioning and reliable
game API that works as intended. Next stop, we start using CV to identify the various objects, lanes
and the player themself to begin the long journey to computing a feature-vector for the game.  

The windows phone link was a problem, I replaced it with Bluestacks emulator (you need to turn off 
anti-virus during download) and it works like a charm, allows me to set it in flight mode so no
game ads. This part of the puzzle has been solved.

### Week 2-6 Oct 2023 ###
