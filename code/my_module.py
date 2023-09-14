import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score, precision_score, recall_score, accuracy_score, roc_auc_score
sns.set_context(context='notebook')

def viz_1(df):
    """
    This function makes a bar graph showing the distribution of respondents with and without a heart condition.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = sns.countplot(data=df, x="Heart_Disease")
    bar_heights = [p.get_height() for p in ax.patches]
    ax.text(0, bar_heights[0], '92%', va='bottom', ha='center', size='large')
    ax.text(1, bar_heights[1], '8%', va='bottom', ha='center', size='large')
    ax.set(ylim=(0, 325000),
           title='Respondents that reported having Heart Disease',
           ylabel='Number of Respondents',
           xlabel='')
    ax.yaxis.set_major_formatter(lambda x, pos: f'{int(x/1000)}K')
    plt.savefig('images/viz_1.jpg',
                bbox_extra_artists=[ax],
                bbox_inches='tight')


def viz_2(df, categorical):
    """
    This function calculates the rate of each binary variable compared to whether or not the person has a heart condition. It then creates a visual to display the distribution. 
    """
    for cat in categorical:
        resp = df[cat].value_counts().index
        nn = (
            (df[cat] == resp[0]) &
            (df['Heart_Disease'] == 'No')).sum() / (df[cat] == resp[0]).sum()
        ny = (
            (df[cat] == resp[0]) &
            (df['Heart_Disease'] == 'Yes')).sum() / (df[cat] == resp[0]).sum()
        yn = (
            (df[cat] == resp[1]) &
            (df['Heart_Disease'] == 'No')).sum() / (df[cat] == resp[1]).sum()
        yy = (
            (df[cat] == resp[1]) &
            (df['Heart_Disease'] == 'Yes')).sum() / (df[cat] == resp[1]).sum()
        num = [nn, yn, ny, yy]

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow([[nn, ny], [yn, yy]], cmap='Blues')
        for i, j in enumerate(num):
            text = str(round(j * 100, 2)) + '%'
            ax.text(i // 2,
                    i % 2,
                    text,
                    ha="center",
                    va="center",
                    color="tab:orange",
                    weight="bold")
        ax.set(xticks=[0, 1],
               xticklabels=['No', 'Yes'],
               xlabel='Heart Condition',
               yticks=[0, 1],
               yticklabels=[resp[0], resp[1]],
               ylabel=cat)
        plt.savefig('images/viz_2.jpg',
                    bbox_extra_artists=[ax],
                    bbox_inches='tight')
        plt.show()


def model_scores(model, X, y, model_list=[], cv=5, model_name=''):
    """
    This is a helper function which takes in a fitted estimator, and cross validates it, calculating the f2-score, accuracy, recall, and roc_auc score. It outputs the summary as a dataframe. It also outputs a list in the case that the table builds on itself. 
    """
    if cv > 1:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=12)
        scoring = {
            'f2': make_scorer(fbeta_score, beta=2),
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, zero_division=0.0),
            'recall': 'recall',
            'roc_auc': 'roc_auc'
        }
        scores = cross_validate(model,
                                X,
                                y,
                                scoring=scoring,
                                cv=skf,
                                n_jobs=-1)
        f2 = round(scores['test_f2'].mean(), 4) * 100
        accuracy = round(scores['test_accuracy'].mean(), 4) * 100
        precision = round(scores['test_precision'].mean(), 4) * 100
        recall = round(scores['test_recall'].mean(), 4) * 100
        roc_auc = round(scores['test_roc_auc'].mean(), 4) * 100
    else:
        y_preds = model.predict(X)
        f2 = round(fbeta_score(y, y_preds, beta=2), 4) * 100
        recall = round(recall_score(y, y_preds), 4) * 100
        accuracy = round(accuracy_score(y, y_preds), 4) * 100
        precision = round(precision_score(y, y_preds, zero_division=0.0),
                          4) * 100
        roc_auc = round(roc_auc_score(y, y_preds), 4) * 100
    model_list.append([model_name, f2, accuracy, precision, recall, roc_auc])
    df = pd.DataFrame(
        model_list,
        columns=['name', 'f2', 'accuracy', 'precision', 'recall', 'roc_auc'])
    return model_list, df


def random_search(X_train, y_train):
    """
    Warning: This may take a long time to run depening on cpu resources.
    This function runs a RandomizedSearchCV through our model's pipeline. It outputs the the scores as a list and data frame.     
    """
    pipe = Pipeline(steps=[('ct', ct),
                           ('xbg',
                            XGBClassifier(random_state=12,
                                          tree_method='hist',
                                          scale_pos_weight=283883 / 24971))])
    params = {
        'xbg__n_estimators': range(50, 1000, 50),
        'xbg__max_depth': range(1, 15),
        'xbg__eta': [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'xbg__colsample_bytree': np.linspace(0, 1, 50),
        'xbg__min_child_weight': range(1, 10),
        'xbg__gamma': [0, 0.1, 1, 10, 100, 1000],
        'xbg__reg_alpha': [0, 0.01, 0.1, 1, 10],
        'xbg__reg_lambda': [0, 0.01, 0.1, 1, 10]
    }
    ftwo_scorer = make_scorer(fbeta_score, beta=2)

    rs = RandomizedSearchCV(estimator=pipe,
                            param_distributions=params,
                            n_iter=500,
                            scoring=ftwo_scorer,
                            n_jobs=-1,
                            cv=5,
                            random_state=12)
    rand_search = rs.fit(X_train, y_train)
    rand_model = rand_search.best_estimator_
    ml5, df5 = model_scores(rand_model,
                            X_train,
                            y_train,
                            model_list=[],
                            cv=5,
                            model_name='Best Estimator from RS')
    return ml5, df5


def grid_search(X_train, y_train):
    """
    Warning: This may take a long time to run depening on cpu resource
    This function runs a GridSearchCV on our pipeline looking for the optimal estimator. It outputs the scores as a list and dataframe.
    """
    xgb = XGBClassifier(reg_lambda=10,
                        reg_alpha=0.1,
                        min_child_weight=1,
                        gamma=10,
                        colsample_bytree=0.42857142857142855,
                        tree_method="hist",
                        scale_pos_weight=283883 / 24971,
                        random_state=12)

    pipe = Pipeline(steps=[('ct', ct), ('xbg', xgb)])

    params = {
        'xbg__n_estimators': np.linspace(400, 800, 10, dtype=int),
        'xbg__max_depth': [2, 3, 4, 5, 6, 7],
        'xbg__eta': np.linspace(0.1, 0.2, 10)
    }

    gs = GridSearchCV(pipe,
                      param_grid=params,
                      scoring=ftwo_scorer,
                      n_jobs=-1,
                      cv=5,
                      verbose=3)

    g_search = gs.fit(X_train, y_train)
    gs_model = g_search.best_estimator_
    ml5, df5 = model_scores(gs_model,
                            X_train,
                            y_train,
                            model_list=ml5,
                            cv=5,
                            model_name='Best Estimator from GS')
    return ml5, df5