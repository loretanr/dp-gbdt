#pragma once


namespace xgboost {
	class Config {
	public:
		//n_estimators : int, optional (default=100). Number of boosted trees to fit.
		int n_estimators = 5;
		//max_depth : int, optional (default=10). Maximum tree depth for base learners, -1 means no limit.
		int max_depth = 6;
		//learning_rate : float, optional (default=0.1). Boosting learning rate.
		float learning_rate = 0.1;
		//min_samples_split : int, optional (default=2). The minimum number of samples required to split an internal node.
		int min_samples_split = 2;
		//min_data_in_leaf : int, optional (default=1). The minimum number of samples required to be at a leaf node.
		int min_data_in_leaf = 1;
		//min_child_weight : float, optional (default=1e-3). Minimum sum of instance weight(hessian) needed in a child(leaf).
		float min_child_weight = 1e-3;
		//colsample_bytree : float, optional (default=1.0). Subsample ratio of columns when constructing each tree.
		float colsample_bytree = 1.0;
		//reg_gamma : float, optional (default=0.0). L1 regularization term on weights.
		float reg_gamma = 0.0;
		//reg_lambda : float, optional (default=0.0). L2 regularization term on weights.
		float reg_lambda = 0.0;
		//max_bin: int or None, optional(default = 225)). Max number of discrete bins for features.
		int max_bin = 100;
	};
}
