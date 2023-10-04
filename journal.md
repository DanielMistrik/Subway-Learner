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
Started work on the score reader. easyocr seems to be the easier 
choice to pick however I think its installation messed with cv2.imshow
as the function no longer works. In any case, score detection does roughly
work but is quite noisy. The feature function will have to do some primitive 
linear projection backed up by the data it gets from the actual reward function
or some kind of outlier detection. Next step is player identification.

For player identification, while I could do the template matching I did with the 
play button I realized that it is too unreliable. Instead I decided to look for a 
unique characteristic in the character, which turned out to be the colour of their 
hoodie. I found the mean of all pixels with the coordinates and it gave me a very reliable
prediction of where the player is. Next up is obstacle identification which I might do 
with template matching of the colouring like with the player.

After playing with some template and OB matching I realized
it wasn't working at all and so came back to the colour matching
which worked like a charm. I also decided to split the screen in 3
parts and the obstacle detection on each as obstacles are always
in one of the 3 equally-sized 'channels'/'lanes' anyways. This allows
me to use the color median method which (by definition) finds only one 
median. The thinking above was on the right track but flawed, the lanes
aren't always equal and especially when the player is in one of the side lanes.
Regardless, I cropped the different screenshots, so it works and now have 5 possible arrays to
take input from as the player moves. It is noisy so the featurization function is gonna have to 
do alot of regularization. Next is to identify the different barriers and middle columns. Then object
detection is complete, and I can start with the featurization function (which I expect is going to be
alot of work).

After doing a lot of tunning I managed to get all the obstacles (except for an ellusive pol I was unable
to encounter) detected. Object detection is done. The color matching is probably the right way to
go but I wouldn't be surprised if we had to tune it as I had to do alot of it myself. Noise is there
but hopefully not terrible. The feature function is going to have to do alot of noise management. 
For the obstacles, the wood colour is very noisy so those will only be classified as obstacles if 
the median for the red paint of obstacle is also sort of near. Next up is to produce the actual
feature function.
