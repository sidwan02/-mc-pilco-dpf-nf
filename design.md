Like there are many GP, will we have many cnf cnns?
Or just one cnf nn for all the states

A: Many nf for each state

We wanna train the nn cnf within reinforce_policy after each apply_policy

We need access to the mean and variance of the states so can we just recompute that from the state or will I need to get that from apply_policy (currenly these lists are not maintained in MC pilco)

When are we actually finding the next_state from the normalizing flows? Is that code implemented?
A: embed the flow prediction in the dynamics model function
when you predict get next satate you should be predicting with the

function called get_next_state -> instead of calling gaussing, sampling and so on is passed mean and variance into forward and updates the next state

How are we going to pretrain the flows? Does it even make sense? For the GP the pretain is fine becausse it does it from the initial state. But we don't have all the states ready (since the cnf nn is training on all the states and their observations aka next states)

input_sequence is a concatenation of the observation from the env (rollback etc) and action
so it has dim 5
if it was just the observation from the environment it would have dim 4
