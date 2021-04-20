import numpy as np
from sklearn import linear_model, svm, metrics

def main():
    # Loading data
    data = np.load("data/data.npz")
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]


    # Setting parameters
    l = x_train.shape[0]
    K_fold = 5
    ratio = 1.0/K_fold

    # Model
    SVM = svm.SVC(max_iter=1e4)
    logistic = linear_model.LogisticRegression(max_iter=1e4)
    models = [SVM, logistic]
    model_name = ["SVM", "LogisticRegression"]

    # Cross validation
    shuffled_indices=np.random.permutation(l)
    accs = [[],[]]
    f1s = [[],[]]
    for i in range(5):
        start=int(l*ratio*i)
        end=int(l*ratio*(i+1))
        print("Iter {}: validation set is [{},{}]".format(i, start, end))
        test_indices =shuffled_indices[start:end]
        train_indices = np.concatenate((shuffled_indices[:start], \
            shuffled_indices[end:]), axis=0)
        for j in range(2):
            models[j].fit(x_train[train_indices], y_train[train_indices])
            acc = models[j].score(x_train[test_indices], y_train[test_indices])
            y_pred = models[j].predict(x_train[test_indices])
            f1 = metrics.f1_score(y_train[test_indices], y_pred, average='binary')
            accs[j].append(acc)
            f1s[j].append(f1)
            print("Iter {}: {}'s CV Accuracy is {}".format(i, model_name[j], acc))

    
    # Test
    print()
    mean_acc = [np.mean(i) for i in accs]
    best = np.argmax(mean_acc)
    acc = models[best].score(x_test, y_test)
    print("According to ACCURACY, {}:{} VS {}:{}".format(model_name[0], mean_acc[0]\
        , model_name[1], mean_acc[1]))
    print("Best Model:", model_name[best])
    print("Testing Accuracy:", acc)
    print()
    mean_f1 = [np.mean(i) for i in f1s]
    best = np.argmax(mean_f1)
    acc = models[best].score(x_test, y_test)
    print("According to F1-SCORE, {}:{} VS {}:{}".format(model_name[0], mean_f1[0] \
        , model_name[1], mean_f1[1]))
    print("Best Model:", model_name[best])
    print("Testing Accuracy:", acc)

if __name__=="__main__":
    main()