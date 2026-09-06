import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# page configuration
st.set_page_config(
    page_title="Scientific Abstract Classification",
    layout="wide"
)

# create outputs directory if not exists
if not os.path.exists('outputs'):
    os.makedirs('outputs')

try:
    nltk.data.find('corpora/stopwords')  # to reach list of ineffective words
except LookupError:
    nltk.download('stopwords')

 # preprocessing and cleaning
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) #  deletes everything except letters and spaces from a to z.
    words = text.split() #  breaks the sentence along the blanks and turns it into a word list.
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words) # converts the list back into sentences.

def train_and_evaluate():
    status_text = st.empty()
    progress_bar = st.progress(0)

    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return None, None

    progress_bar.progress(10)

    train_df['clean_text'] = train_df['Comment'].apply(preprocess_text)
    test_df['clean_text'] = test_df['Comment'].apply(preprocess_text)
    
    y_train = train_df['Topic']
    y_test = test_df['Topic']
    
    progress_bar.progress(30)

    # vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_df['clean_text'])
    X_test = tfidf.transform(test_df['clean_text'])
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    
    progress_bar.progress(50)

    # training models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, metric='cosine')
    }
    
    results = {}
    reports = {}
    best_acc = 0
    best_model_name = ""
    best_model_obj = None
    sample_preds = {}
    all_predictions = {}
    
    total_models = len(models)
    current_model = 0
    
    for name, model in models.items():
        current_model += 1
        status_text.text(f"Training {name}...")
        
        # train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_predictions[name] = y_pred
        
        # evaluate
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        # generate detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        reports[name] = report
        
        # generate confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.savefig(f'outputs/confusion_matrix_{name.replace(" ", "_")}.png')
        plt.close()
        
        # check best model
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model_obj = model
            
        progress_bar.progress(50 + int((current_model / total_models) * 40))

    if best_model_obj:
        joblib.dump(best_model_obj, 'best_model.pkl')

    # comparison plot
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.savefig('outputs/model_comparison.png')
    plt.close()
    
    # create one consolidated sample set ->10 random samples
    consolidated_samples = test_df.copy()
    consolidated_samples['True Label'] = y_test
    
    # add predictions for each model
    for model_name, preds in all_predictions.items():
        consolidated_samples[model_name] = preds
        
    # pick 10 random samples
    sample_preds = consolidated_samples.sample(n=10, random_state=99)

    progress_bar.progress(100)
    status_text.text("Training completed successfully!")
    
    return results, reports, best_model_name, sample_preds, all_predictions, test_df

# --main app structure--

st.title("Scientific Abstract Classification Project")

# Tabs
tab_run, tab_charts, tab_logs = st.tabs(["Run & Report", "Training Graphs", "Full Prediction Logs"])

# -- tab 1: Run & Report --
with tab_run:
    st.header("Execute Training Pipeline")
    st.markdown("Click the button to train models and view detailed classification metrics (Precision, F1-Score, Recall).")
    
    if st.button("Start Training", type="primary"):
        with st.spinner("Training in progress..."):
            results, reports, best_model, sample_preds, all_predictions, test_df_global = train_and_evaluate()
            #storing in session state,to keep the data alive in memory
            st.session_state['results'] = results
            st.session_state['reports'] = reports
            st.session_state['best_model'] = best_model
            st.session_state['sample_preds'] = sample_preds
            st.session_state['all_predictions'] = all_predictions
            st.session_state['test_df_global'] = test_df_global

    if 'results' in st.session_state:
        results = st.session_state['results']
        reports = st.session_state['reports']
        best_model = st.session_state['best_model']
        sample_preds = st.session_state['sample_preds']
        

        if True:
            
            if results and reports:
                #  to increase metric label font size
                st.markdown("""
                    <style>
                    [data-testid="stMetricLabel"] {
                        font-size: 25px !important;
                        font-weight: 700 !important;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.subheader("Model Performance Overview")
                
                # create 3 columns for metrics
                m1, m2, m3 = st.columns(3)
                models_list = ['Naive Bayes', 'Decision Tree', 'KNN']
                cols_list = [m1, m2, m3]
                
                for idx, model_name in enumerate(models_list):
                    acc = results[model_name]
                    # highlight if it's the best model
                    if model_name == best_model:
                        cols_list[idx].metric(label=f"{model_name}", value=f"{acc:.4f}", delta="Best Model", delta_color="normal")
                    else:
                        cols_list[idx].metric(label=f"{model_name}", value=f"{acc:.4f}")
                
                st.success(f"**Best Model Selected:** {best_model} with Accuracy: **{results[best_model]:.4f}**")               
                st.divider()
                st.subheader("Detailed Performance Matrix")
                
                combined_rows = []
                target_classes = ['Biology', 'Chemistry', 'Physics']
                
                # class rows
                for cls in target_classes:
                    row_data = {'Class': cls}
                    for model_name in models_list:
                        if model_name in reports:                            
                            row_data[f"{model_name} F1"] = reports[model_name][cls]['f1-score']  # class-specific f1 score & recall 
                            row_data[f"{model_name} Recall"] = reports[model_name][cls]['recall']
                    combined_rows.append(row_data)
                
                df_combined = pd.DataFrame(combined_rows)

                sorted_cols = ['Class']
                for model_name in models_list:
                    sorted_cols.append(f"{model_name} F1")
                    sorted_cols.append(f"{model_name} Recall")
                df_combined = df_combined[sorted_cols]
                
                # custom css to make headers and index brighter
                st.markdown("""
                    <style>
                        /* Target all table headers within the dataframe */
                        [data-testid="stDataFrame"] th,
                        [data-testid="stDataFrame"] .col_heading,
                        [data-testid="stDataFrame"] .index_name,
                        [data-testid="stDataFrame"] .blank {
                            background-color: #2c2c2c !important;
                            color: #ffffff !important;
                            font-size: 15px !important;
                            font-weight: 900 !important; /* Extra bold */
                            opacity: 1 !important;
                            fill: white !important; /* For SVGs if any */
                        }
                        
                        /* Target the specific text spans inside headers if they exist */
                        [data-testid="stDataFrame"] th div,
                        [data-testid="stDataFrame"] th span {
                            color: #ffffff !important;
                            opacity: 1 !important;
                        }

                        /* Data Cells */
                        [data-testid="stDataFrame"] tbody td {
                            color: #ffffff !important;
                            font-weight: bold !important;
                            font-size: 14px !important;
                        }
                    </style>
                """, unsafe_allow_html=True)
                
                # display with color gradient styling
                st.dataframe(
                    df_combined.style
                    .format("{:.4f}", subset=sorted_cols[1:]) 
                    .background_gradient(cmap='Greens', subset=sorted_cols[1:], vmin=0.5, vmax=1.0)
                    .set_properties(**{'font-weight': 'bold', 'font-size': '15px'}, subset=['Class']),
                    use_container_width=True,
                    hide_index=True
                )

                st.divider()
                
                #consolidated sample predictions
                st.subheader("Sample Predictions Comparison (10 Random Samples)")
                st.markdown("Side-by-side comparison of how each model classified the same 10 random examples.")
                h1, h2, h3, h4, h5 = st.columns([6, 1.2, 1.2, 1.2, 1.2])
                h1.markdown("**Abstract Text**")
                h2.markdown("**True Class**")
                h3.markdown("**Naive Bayes**")
                h4.markdown("**Decision Tree**")
                h5.markdown("**KNN**")
                st.markdown("---")

                # data rows
                for _, row in sample_preds.iterrows():
                    c1, c2, c3, c4, c5 = st.columns([6, 1.2, 1.2, 1.2, 1.2])

                    c1.write(row['Comment'])
                    c2.info(row['True Label'])
                    nb_pred = row['Naive Bayes']
                    if nb_pred == row['True Label']:
                        c3.success(nb_pred)
                    else:
                        c3.error(nb_pred)

                    dt_pred = row['Decision Tree']
                    if dt_pred == row['True Label']:
                        c4.success(dt_pred)
                    else:
                        c4.error(dt_pred)

                    knn_pred = row['KNN']
                    if knn_pred == row['True Label']:
                        c5.success(knn_pred)
                    else:
                        c5.error(knn_pred)        
                    st.markdown("---")

# -- tab 2: Training Graphs --
with tab_charts:
    st.header("Visual Performance Metrics")
    
    # check if files exist (if training was run previously)
    if os.path.exists('outputs/model_comparison.png'):
        
        # 1) accuracy chart
        st.subheader("1. Accuracy Comparison")
        st.image('outputs/model_comparison.png', caption="Accuracy Comparison Chart")
        
        st.divider()
        
        # 2) confusion matrices
        st.subheader("2. Confusion Matrices")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Naive Bayes**")
            if os.path.exists('outputs/confusion_matrix_Naive_Bayes.png'):
                st.image('outputs/confusion_matrix_Naive_Bayes.png')
        with c2:
            st.markdown("**Decision Tree**")
            if os.path.exists('outputs/confusion_matrix_Decision_Tree.png'):
                st.image('outputs/confusion_matrix_Decision_Tree.png')
        with c3:
            st.markdown("**KNN**")
            if os.path.exists('outputs/confusion_matrix_KNN.png'):
                st.image('outputs/confusion_matrix_KNN.png')
        
        st.divider()

        # 3) performance analysis
        st.subheader("3. Performance Analysis")
        st.info("""
        **Key Takeaways from the Confusion Matrices:**
        
        *   **Naive Bayes Dominance:** As clearly seen in the matrices, **Naive Bayes** exhibits the darkest diagonal blocks and the cleanest off-diagonal areas. This confirms it is the best-suited model for this text classification task, effectively handling the high-dimensional TF-IDF features.
        
        *   **Biology Performance:** Although the Biology class has the highest number of samples in the test set (which naturally increases raw counts), it also consistently achieves the **highest F1-Scores** across all models. This confirms that it is indeed the most well-defined class, and the high success rate is not just due to sample size but due to better separability of its terminology.
        
        *   **Physics-Chemistry Overlap:** The generated matrices often show slight confusion between **Physics** and **Chemistry**. This is expected, as they share many common terms (e.g., *atom, energy, electron*), causing models like the Decision Tree to occasionally struggle in distinguishing them.
        
        *   **Decision Tree Struggles:** The Decision Tree matrix appears more scattered compared to Naive Bayes. This indicates it struggles to create simple decision rules for complex text data, leading to a lower overall accuracy.
        """)
    else:
        st.info("No charts found yet. Please go to the 'Run & Report' tab and click 'Start Training' first.")


# -- tab 3: full prediction logs --
with tab_logs:
    st.header("Full Prediction Logs")
    st.markdown("This section allows you to explore the complete test dataset with full text visibility.")
    st.caption("🟦 **Blue**: True Label | 🟩 **Green**: Correct Prediction | 🟥 **Red**: Incorrect Prediction")

    # check if we have the data
    if 'all_predictions' in st.session_state and 'test_df_global' in st.session_state:
        all_preds = st.session_state['all_predictions']
        test_df_global = st.session_state['test_df_global']
        
        # prepare data
        master_df = test_df_global[['Comment', 'Topic']].copy()
        master_df.rename(columns={'Topic': 'True Label', 'Comment': 'Abstract Text'}, inplace=True)
        for model_name, preds in all_preds.items():
            master_df[f"{model_name} Pred"] = preds
            
        # pagination settings
        items_per_page = 20
        total_items = len(master_df)
        total_pages = (total_items // items_per_page) + (1 if total_items % items_per_page > 0 else 0)
        
        c_page, c_info = st.columns([1, 4])
        with c_page:
            page = st.number_input("Page Number", min_value=1, max_value=total_pages, value=1, step=1)
        with c_info:
            st.markdown(f"**Showing {(page-1)*items_per_page + 1} - {min(page*items_per_page, total_items)} of {total_items} samples**")
        
        st.divider()
        
        # slicing
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        subset_df = master_df.iloc[start_idx:end_idx]
        
        # headers
        h1, h2, h3, h4, h5 = st.columns([6, 1.2, 1.2, 1.2, 1.2])
        h1.markdown("**Abstract Text**")
        h2.markdown("**True**")
        h3.markdown("**Naive Bayes**")
        h4.markdown("**Decision Tree**")
        h5.markdown("**KNN**")
        st.markdown("---")

        # rendering rows
        for idx, row in subset_df.iterrows():
            c1, c2, c3, c4, c5 = st.columns([6, 1.2, 1.2, 1.2, 1.2])
            
            # 1) text
            c1.write(row['Abstract Text'])
            
            # 2) true label (blue)
            c2.info(row['True Label'])
            
            # 3) naive bayes (green/red)
            nb_pred = row['Naive Bayes Pred']
            if nb_pred == row['True Label']:
                c3.success(nb_pred)
            else:
                c3.error(nb_pred)
                
            # 4) decision tree (green/red)
            dt_pred = row['Decision Tree Pred']
            if dt_pred == row['True Label']:
                c4.success(dt_pred)
            else:
                c4.error(dt_pred)
                
            # 5) knn (green/red)
            knn_pred = row['KNN Pred']
            if knn_pred == row['True Label']:
                c5.success(knn_pred)
            else:
                c5.error(knn_pred)
            
            st.markdown("---")
            
    else:
        st.info("No logs available. Please go to the 'Run & Report' tab and click 'Start Training' first.")


