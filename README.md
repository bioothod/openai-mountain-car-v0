# openai-mountain-car-v0
OpenAI challenges repository

I tested various approaches and found that properly tuned DQN plus cross-entropy pool solves
this problem in the fastest way.

By DQN+CE I mean common DQN technique, but batches sampled each time for experience reply
are selected proportionally to how good their appropriate episode was compared to
the worst one with -200 total reward.

In common cross-entropy we basically select the best episodes and learn network
to correctly predict action based on those steps. This drops experience for the
wrong/non-existing steps and actions, which might be good to learn too.

Another approach I tested was main/follower networks, i.e. when you 'read'
from the follower network which is slowly follows main network used for learning.
This does not really change convergence noticeably,
maybe reduces oscillations, which always happen likely because of overfitting.

Another approach was a3c and actor-critic model. This never reached performance of DQN
and more commonly rises serious questions about its applicability for this task,
since there is no clear convergence logic and proper reward match - it is always possible
to create a set of steps which will have the same discounted reward, but will not lead
to the winning strategy.

DQN is a simple network consisting of 3 layers: 50, 190 and 3 neurons, the first two have biases.
I use `tanh` nonlinearity, `relu` never congested and in my opinion it is very overvalued function.
Tanh has a nice property of being positive and negative, which is useful in this problem.
The last layer uses linear activation function since Q-values are large enough.

I use l1+l2 regularization and RMSProp optimizer.

By tuning number of neurons you can heavily affect performance,
so I believe my numbers can further be improved.

Code for all mentioned ideas is included in the repository
https://github.com/bioothod/openai-mountain-car-v0,
it is implemented in pure python + TF. To run the winning solution just type
```
$ python mc0.py
```
It will create `mc0` and `mc0_wrappers` directories in the current dir, the former can be used
