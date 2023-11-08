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

Added a is_alive tracker and the score function was way too noisy
and (computationally-cheap) smoothing was not working well either and 
as the score is roughly a monotonically increasing linear function I 
reimplemented such a thing for the score function, this reduces computational
cost by removing the easyocr library which was expensive. Working on the feature function.
The challenge was parsing the cv output from the view obstacles function
but I got the first part of the state vector to work with a quite satisfying 
accuracy. 

Finally finished the feature function. I had to remove the wall
states as they were too inaccurate. Otherwise, some tweaking to the 
pixel distances worked fine. The current approach normalizes distance to
an object from 2 (not visible) then [1,0] where 1 is the furthest visible location
and 0 is collision with the obstacle. Now that the feature function is
done I can move on to set-up for RL. The state function will probably
have to poll the state function 3 times and average to remove some noise.

Started work on the set-up for RL. Best approach is to use the openAI gym API and for this I have
to implement a relatively simple environment class. Working on the feature vector function which
will poll the game state 3 times and normalize to remove noise. Will then test the class out to see 
if it is working as intended. Then it will be time to run some RL algos (finally!).


### Weekend 7-8 Oct 2023 ###
Finished the normalizing feature vectors. Intentionally ignored cases with a None player location and 
minimized the platform elements because of how sensitive it is. Now we proceed to test it and then 
apply the first RL algos.

Worked on fixing the errors with the env implementation. Main ones were casting
and the get_feature_vector() function which still throw some warnings I am going over.

### Week 9-13 Oct 2023 ###
Finished the bugs that were preventing smooth runs on the random learner. I will now try some
stable-baseline 3 algos to see how well they learn (starting with the very
general DQN). After letting the random algo run a couple of times the median 
score hovers around 8-10 while the best was 39. I will consider anything above 50 a significant first
step. Moved the entire thing to gymnasium and did some feature vector cleaning
so the baseline RF algos would run. About to run the first solid DQN on 100 timesteps, a big day.
100 timesteps successfully trained although the resulting learner was really poor with an 
average score of 2.8-3. I will train overnight with 10,000 timesteps.

Trained overnight on 10,000 timesteps and no difference. The learner still
always reverts to going left all the time and losing almost immediately. The number of timesteps 
had no impact on performance, it seems the choice of algorithm was wrong. Before switching algorithms
I will make the reward function more intelligent and through it push some basic rules of the game
(i.e. don't turn right on the right lane, dont duck for a train). This reward set-up seems to have some
initial positive feedback with the learner moving out of the train's direction and not
just running left.

### Week Oct 22-29 2023 ###
Looking over the DQN code one of the reasons for the extremely poor performance was the 
learning_starts parameter was set at 50,000 steps which we never reached, this was dramatically reduced
with the learning_rate and batch_sized increased for more exploration and stability. Attempts weren't
very successful. I need to greatly modify the reward function so its simpler and more clear about
what we want and also convert to Deep Q-Learning with the entire screen as input and a CNN as the
agent as the current approach isn't working well.

### Weekend Oct 28-29 2023 ###
Unfortunately the CNN DQN performed equally poorly with more than 
30,000 timesteps of learning. Clearly the search space is too big so we will need
to do some pre-training with behavioral cloning.

Really good progress on the DAgger/Behvaioral Cloning Front. I got an 
initial version up and running and it seems to be faring quite well. There are some weird
bugs arising from step_wait() which I need to debug, it could be a me issue but the bug is very
weird so might be something on their side. In any case Ill solve that tomorrow.

### Weekend Nov 4-5 2023 ###
Playtime duration as an average of 50 games: 
BC-Learner 14.64, DAgger 20.89, QE-DAgger 17.89, QE-DAgger2 16.39
Note BC learner was trained on 2000 samples, DAgger on ~600 Samples and 
QE-DAgger & QE-DAgger-2 on 500 samples. Clearly DAgger is the way to go.
I did QE-DAgger at a 1000 samples with a modified weighted loss function and 
it ran for 11.55 seconds. Clearly not the way to go. I will add another 
convolutional layer to the CNN, put back the old loss function and run DAgger for 2000 samples.