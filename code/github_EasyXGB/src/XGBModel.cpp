#include "XGBModel.h"

XGBModel::XGBModel(Dataset* dataset, ModelParam params):
    dataset(dataset), params(params) {
}

void XGBModel::train() {
    Timer timer;
    timer.start();
    float train_avg_rmse = 0, train_avg_acc = 0;
    float valid_avg_rmse = 0, valid_avg_acc = 0;
    for (int i = 0; i < params.num_folds; ++i) {
        cout << endl << "********** Fold " << i+1 << " *********" << endl;
        train_fold(i, train_avg_rmse, train_avg_acc, valid_avg_rmse, valid_avg_acc);
    }

    cout << endl << params.num_folds << " Folds " << "avg results: " << endl;
    if (params.objective == "regression") {
        cout << "train avg rmse: " << train_avg_rmse << ", valid avg rmse: " << valid_avg_rmse << endl;
    } else {
        cout << "train avg acc: " << train_avg_acc << ", valid avg acc: " << valid_avg_acc << endl;
    }
    timer.stop();
    cout << params.num_folds << " Folds training using total time: " << timer.count() << "s" << endl;
}

void XGBModel::train_fold(int fold_index, float& train_avg_rmse, float& train_avg_acc,
                          float& valid_avg_rmse, float& valid_avg_acc) {
    Timer timer;
    timer.start();
    trees.clear();
    dataset->switch_new_fold(fold_index, params.num_folds);
    dataset->reset_preds();
    for (int i = 0; i < params.num_trees; ++i) {
        cout << "========== Fold " << fold_index+1;
        cout <<  ", Tree " << i+1 << " =============" << endl;
        shared_ptr<Tree> tree = make_shared<Tree>(dataset, params);
        tree->grow_tree();
        if (params.prune_tree) {
            tree->prune_tree();
        }
        tree->inference_tree(params.learning_rate);
        if (params.print_tree) {
            tree->print_tree();
        }

        if (params.objective == "regression") {
            cout << "Train RMSE: " << dataset->rmse("train") << endl;
        } else {
            cout << "Train ACC: " << dataset->acc("train") << endl;
        }
        if (params.objective == "regression") {
            cout << "Valid RMSE: " << dataset->rmse("valid") << endl;
        } else {
            cout << "Valid ACC: " << dataset->acc("valid") << endl;
        }
        trees.push_back(tree);
    }

    train_avg_rmse += dataset->rmse("train") / params.num_folds;
    train_avg_acc += dataset->acc("train") / params.num_folds;
    valid_avg_rmse += dataset->rmse("valid") / params.num_folds;
    valid_avg_acc += dataset->acc("valid") / params.num_folds;

    timer.stop();
    cout << "Train Fold using time: " << timer.count() << "s" << endl;
}

void XGBModel::evaluate(Dataset* dataset) {
    dataset->reset_preds();
    for (int i = 0; i < dataset->num_instances(); ++i) {
        dataset->set_pred(i, inference(dataset->get(i)));
    }
    cout << "Evaluate RMSE: " << dataset->rmse("test") << endl;
    cout << "Evaluate ACC: " << dataset->acc("test") << endl;
}

float XGBModel::inference(vector<float>& instance) {
    float pred = 0.0;
    for (auto tree: trees) {
        pred += params.learning_rate * tree->inference_tree(instance);
    }
    return pred;
}
