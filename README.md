# threshold_estimation
Carry out an estimate of the position of a structural break in data. This uses a full grid search and so is not greedy in the same way a tree regression is - although it is computationally more expensive. This approach also allows for p tests a la Bai Perron (1998), against a null of no or one-fewer structural breaks.
