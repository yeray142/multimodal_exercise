import pandas as pd
import numpy as np


def compute_avg_acc(pred_gt):
    acc = 0
    for c, f in enumerate(pred_gt['VideoName']):
        #print(c,f,pred_gt.iloc[c]['ground_truth'],pred_gt.iloc[c]['prediction'])
        if(pred_gt.iloc[c]['prediction'] == pred_gt.iloc[c]['ground_truth']):
            acc +=1
    
    avg_acc = acc/len(pred_gt)
    print("----")
    print("Global Average accuracy = %.4f" % (avg_acc))
    print("----")
    return avg_acc


def compute_avg_acc_per_age_cat(pred_gt):
    acc = [0, 0, 0, 0, 0, 0, 0]
    age_distribution = [0, 0, 0, 0, 0, 0, 0]
    for c, f in enumerate(pred_gt['VideoName']):
        # getting the age category
        cat_idx = pred_gt.iloc[c]['ground_truth']-1

        # computing the age distribution
        age_distribution[cat_idx] = age_distribution[cat_idx]+1

        # computing the accuracy
        if(pred_gt.iloc[c]['prediction'] == pred_gt.iloc[c]['ground_truth']):
            acc[cat_idx] = acc[cat_idx]+1

    print("Age distribution = ", age_distribution)
    avg_acc_per_cat = []
    for c in range(len(acc)):
        avg_acc_per_cat.append(acc[c] / age_distribution[c])
        print("Average accuracy of age category %d = %.4f" % (c+1,avg_acc_per_cat[c]))
    print("----")
    return avg_acc_per_cat


def compute_avg_acc_per_gender_cat(pred_gt,full_annotations):
    # Gender: Male=1, Female=2
    acc = [0, 0]
    gender_distribution = [0, 0]
    for c, f in enumerate(pred_gt['VideoName']):
        # getting the index of this sample in 'full_annotations'
        idx = full_annotations[full_annotations['VideoName']==f].index.values[0]
        
        # getting the gender category
        cat_idx = full_annotations.iloc[idx]['Gender']-1
        
        # computing the gender distribution
        gender_distribution[cat_idx] = gender_distribution[cat_idx]+1

        # computing the accuracy
        if(pred_gt.iloc[c]['prediction'] == pred_gt.iloc[c]['ground_truth']):
            acc[cat_idx] = acc[cat_idx]+1

    print("Gender distribution (Male | Female) = ", gender_distribution)
    avg_acc_per_cat = []
    for c in range(len(acc)):
        avg_acc_per_cat.append(acc[c] / gender_distribution[c])
        if(c==0):
            print("Average accuracy of Male category = %.4f" % (avg_acc_per_cat[c]))
        else:
            print("Average accuracy of Female category = %.4f" % (avg_acc_per_cat[c]))
    print("----")
    return avg_acc_per_cat


def compute_avg_acc_per_ethnicity_cat(pred_gt,full_annotations):
    # Ethnicity: Asian=1, Caucasian=2, African-American=3
    acc = [0, 0, 0]
    ethnicity_distribution = [0, 0, 0]
    for c, f in enumerate(pred_gt['VideoName']):
        # getting the index of this sample in 'full_annotations'
        idx = full_annotations[full_annotations['VideoName']==f].index.values[0]

        # getting the ethnicity category
        cat_idx = full_annotations.iloc[idx]['Ethnicity']-1

        # computing the gender distribution
        ethnicity_distribution[cat_idx] = ethnicity_distribution[cat_idx]+1

        # computing the accuracy
        if(pred_gt.iloc[c]['prediction'] == pred_gt.iloc[c]['ground_truth']):
            acc[cat_idx] = acc[cat_idx]+1

    print("Ethnicity distribution (Asian | Caucasian | African-American) = ", ethnicity_distribution)
    avg_acc_per_cat = []
    for c in range(len(acc)):
        avg_acc_per_cat.append(acc[c] / ethnicity_distribution[c])
        if(c==0):
            print("Average accuracy of Asian category = %.4f" % (avg_acc_per_cat[c]))
        if(c==1):
            print("Average accuracy of Caucasian category = %.4f" % (avg_acc_per_cat[c]))
        if(c==2):
            print("Average accuracy of African-American category = %.4f" % (avg_acc_per_cat[c]))
    print("----")
    return avg_acc_per_cat


def compute_bias_metric(avg_acc_per_age_cat,bias_id):
    print("--- %s ---" % (bias_id))
    print(avg_acc_per_age_cat)

    d = []
    for i in range(0,len(avg_acc_per_age_cat)):
        for j in range(0,len(avg_acc_per_age_cat)):
            if(i<j):
                d.append(np.abs(avg_acc_per_age_cat[i]-avg_acc_per_age_cat[j]))
                print("acc[%.4f]-acc[%.4f] = %.4f" % (avg_acc_per_age_cat[i],avg_acc_per_age_cat[j],d[-1]))

    bias = np.mean(d)
    print("%s = %.4f" % (bias_id, bias))
    return bias


def main(pred_gt,full_annotations):

    # global average accuracy
    avg_acc = compute_avg_acc(pred_gt)

    # average accuracy per age category
    avg_acc_per_age_cat = compute_avg_acc_per_age_cat(pred_gt)

    # average accuracy per gender category
    avg_acc_per_gender_cat = compute_avg_acc_per_gender_cat(pred_gt,full_annotations)

    # average accuracy per ethnicity category
    avg_acc_per_ethnicity_cat = compute_avg_acc_per_ethnicity_cat(pred_gt,full_annotations)

    # computing our bias metric (mean average accuracy among all categories within each group)
    b_a = compute_bias_metric(avg_acc_per_age_cat,'Age bias')
    b_g = compute_bias_metric(avg_acc_per_gender_cat,'Gender bias')
    b_e = compute_bias_metric(avg_acc_per_ethnicity_cat,'Ethnicity bias')

    print("=======")
    print("Average bias = %.4f" % ((b_a+b_g+b_e)/3))
    print("=======")

if __name__ == '__main__':
    full_annotations = pd.read_csv(r'test_set_age_labels.csv',sep = ',')
    pretictions_with_gt = pd.read_csv(r'predictions_test_set.csv',sep = ',')

    main(pretictions_with_gt, full_annotations)

