Like there are many GP, will we have many cnf cnns?
Or just one cnf nn for all the states

We wanna train the nn cnf within reinforce_policy after each apply_policy

We need access to the mean and variance of the states so can we just recompute that from the state or will I need to get that from apply_policy (currenly these lists are not maintained in MC pilco)

When are we actually finding the next_state from the normalizing flows? Is that code implemented?

How are we going to pretrain the flows? Does it even make sense? For the GP the pretain is fine becausse it does it from the initial state. But we don't have all the states ready (since the cnf nn is training on all the states and their observations aka next states)
