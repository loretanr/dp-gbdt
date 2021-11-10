#ifndef BDD47A93_27B1_403C_8BE2_AC0FF3C04F6B
#define BDD47A93_27B1_403C_8BE2_AC0FF3C04F6B

struct sgx_dataset {
    double *X;
    double *y;
    unsigned num_rows;
    unsigned num_cols;
    char *name;
};

struct sgx_modelparams {
    int nb_trees;
    double privacy_budget;
    unsigned use_dp;
    unsigned gradient_filtering;
    unsigned balance_partition;
    unsigned leaf_clipping;
    unsigned scale_y;
    char *task;
    unsigned task_len;
    int *num_idx;
    unsigned num_idx_len;
    int *cat_idx;
    unsigned cat_idx_len;
};

#endif /* BDD47A93_27B1_403C_8BE2_AC0FF3C04F6B */
