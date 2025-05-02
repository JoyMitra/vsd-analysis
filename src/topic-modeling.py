import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
# import nltk
# from nltk.corpus import stopwords
# from tqdm import tqdm
import pickle
import logging
from datetime import datetime

CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS) + ['__scenario', 'contentfeature', 'contenttype', 'testing', 'screenshot', 'coverage', 'assets', 'asset', 'attachment', 'attachment', 'width', 'am', 'pm', 'img', 'alt', 'src', 'react', 'mongodb', 'html', 'js', 'ts', 'tsx', 'br', 'examples', 'example', 'scenarios', 'scenario', 'given', 'want', 'based', 'feature', 'features']

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('topic_modeling.log'),
        logging.StreamHandler()
    ]
)

# def setup_nltk():
#     """Download required NLTK resources"""
#     try:
#         nltk.download('stopwords', quiet=True)
#         nltk.download('punkt', quiet=True)
#         logging.info("NLTK resources downloaded successfully")
#     except Exception as e:
#         logging.error(f"Error downloading NLTK resources: {e}")
#         raise

def read_text_files(main_dir):
    """
    Read all text files from all subdirectories of the main directory
    
    Args:
        main_dir (str): Path to the main directory containing subdirectories with txt files
        
    Returns:
        tuple: (texts, metadata) where texts is a list of document contents and 
               metadata is a DataFrame with document info
    """
    all_texts = []
    metadata = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(main_dir):
        txt_files = [f for f in files if f.endswith('.txt')]
        
        for txt_file in txt_files:
            file_path = os.path.join(root, txt_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    
                    # Basic preprocessing
                    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
                    text = text.strip()
                    
                    if len(text) > 0:  # Only include non-empty documents
                        all_texts.append(text)
                        
                        # Get relative path from main directory
                        rel_path = os.path.relpath(root, main_dir)
                        
                        metadata.append({
                            'directory': rel_path,
                            'filename': txt_file,
                            'filepath': file_path,
                            'length': len(text)
                        })
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
    
    logging.info(f"Processed {len(all_texts)} text files in total")
    return all_texts, pd.DataFrame(metadata)

def preprocess_texts(texts, max_features=5000, max_df=10, min_df=1):
    """
    Preprocess texts and create document-term matrix
    
    Args:
        texts (list): List of document texts
        max_features (int): Maximum number of features for the vectorizer
        max_df (float): Ignore terms that appear in more than this fraction of documents
        min_df (int): Ignore terms that appear in fewer than this number of documents
        
    Returns:
        tuple: (dtm, vectorizer) where dtm is the document-term matrix and 
               vectorizer is the fitted CountVectorizer
    """
    logging.info("Preprocessing texts and creating document-term matrix")
    
    # Create and configure the vectorizer
    vectorizer = CountVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        stop_words=CUSTOM_STOP_WORDS,
    )
    
    # Create document-term matrix
    dtm = vectorizer.fit_transform(texts)
    
    logging.info(f"Created document-term matrix with shape {dtm.shape}")
    return dtm, vectorizer

def run_topic_modeling(dtm, n_topics=10, max_iter=10, random_state=42):
    """
    Run LDA topic modeling on the document-term matrix
    
    Args:
        dtm: Document-term matrix
        n_topics (int): Number of topics to extract
        max_iter (int): Maximum number of iterations for LDA
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (lda_model, doc_topic_dist) where lda_model is the fitted LDA model and
               doc_topic_dist is the document-topic distribution
    """
    logging.info(f"Running LDA with {n_topics} topics")
    
    # Create and fit LDA model
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method='online',
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    # Transform documents to topic space
    doc_topic_dist = lda_model.fit_transform(dtm)
    
    logging.info("LDA model fitting completed")
    return lda_model, doc_topic_dist

def create_visualizations(texts, metadata, lda_model, dtm, vectorizer, doc_topic_dist, output_dir, n_topics):
    """
    Create and save various visualizations of the topic model
    
    Args:
        texts (list): List of document texts
        metadata (DataFrame): DataFrame with document metadata
        lda_model: Fitted LDA model
        dtm: Document-term matrix
        vectorizer: Fitted CountVectorizer
        doc_topic_dist: Document-topic distribution
        output_dir (str): Directory to save visualizations
        n_topics (int): Number of topics
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Topic keywords visualization
    visualize_topic_keywords(lda_model, vectorizer, output_dir, n_topics)
    
    # 2. Topic distribution across documents
    visualize_doc_topics(doc_topic_dist, metadata, output_dir)
    
    # 3. Word clouds for each topic
    create_topic_wordclouds(lda_model, vectorizer, output_dir, n_topics)
    
    # 4. Topic similarity visualization
    visualize_topic_similarity(lda_model, output_dir)
    
    # 5. Document clustering by topic
    visualize_document_clustering(doc_topic_dist, output_dir)

def visualize_topic_keywords(lda_model, vectorizer, output_dir, n_topics, n_top_words=15):
    """Create and save a visualization of top keywords for each topic"""
    feature_names = vectorizer.get_feature_names_out()
    
    # Create figure
    fig, axes = plt.subplots(n_topics // 2 + n_topics % 2, 2, figsize=(15, n_topics * 2))
    axes = axes.flatten()
    
    # For each topic, plot the top n_top_words keywords
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        
        ax = axes[topic_idx]
        ax.barh(top_features, weights)
        ax.set_title(f'Topic {topic_idx + 1}', fontdict={'fontsize': 15})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_keywords.png'), dpi=200)
    logging.info("Saved topic keywords visualization")

def visualize_doc_topics(doc_topic_dist, metadata, output_dir):
    """Create and save a visualization of topic distribution across documents"""
    # Add dominant topic to metadata
    metadata = metadata.copy()
    metadata['dominant_topic'] = np.argmax(doc_topic_dist, axis=1) + 1
    
    # Create figure
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(x='dominant_topic', data=metadata)
    ax.set_title('Distribution of Dominant Topics Across Documents', fontsize=16)
    ax.set_xlabel('Topic Number', fontsize=14)
    ax.set_ylabel('Number of Documents', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'doc_topic_distribution.png'), dpi=200)
    logging.info("Saved document-topic distribution visualization")

def create_topic_wordclouds(lda_model, vectorizer, output_dir, n_topics, max_words=100):
    """Create and save word clouds for each topic"""
    feature_names = vectorizer.get_feature_names_out()
    
    # Create directory for word clouds if it doesn't exist
    wordcloud_dir = os.path.join(output_dir, 'wordclouds')
    if not os.path.exists(wordcloud_dir):
        os.makedirs(wordcloud_dir)
    
    # Create a word cloud for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        # Get the weights and words for this topic
        word_dict = {feature_names[i]: topic[i] for i in topic.argsort()[:-max_words-1:-1]}
        
        # Generate the word cloud
        wordcloud = WordCloud(
            width=800, 
            height=800, 
            background_color='white',
            max_words=max_words
        ).generate_from_frequencies(word_dict)
        
        # Save the word cloud
        plt.figure(figsize=(8, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_idx + 1}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(wordcloud_dir, f'topic_{topic_idx + 1}_wordcloud.png'), dpi=200)
        plt.close()
    
    logging.info(f"Saved {n_topics} topic word clouds")

# def visualize_topic_similarity(lda_model, output_dir):
#     """Create and save a visualization of topic similarity"""
#     # Compute topic similarity matrix
#     n_topics = lda_model.n_components
#     topic_similarity = np.zeros((n_topics, n_topics))
    
#     for i in range(n_topics):
#         for j in range(n_topics):
#             # Compute cosine similarity between topic vectors
#             similarity = np.dot(lda_model.components_[i], lda_model.components_[j])
#             similarity /= (np.linalg.norm(lda_model.components_[i]) * np.linalg.norm(lda_model.components_[j]))
#             topic_similarity[i, j] = similarity
    
#     # Create figure
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         topic_similarity, 
#         annot=True, 
#         fmt='.2f', 
#         cmap='YlGnBu',
#         cbar_kws={'label': 'Cosine Similarity'},
#         xticklabels=[f'Topic {i+1}' for i in range(n_topics)],
#         yticklabels=[f'Topic {i+1}' for i in range(n_topics)]
#     )
#     plt.title('Topic Similarity Matrix', fontsize=16)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'topic_similarity.png'), dpi=200)
#     logging.info("Saved topic similarity visualization")

def visualize_topic_similarity(lda_model, output_dir):
    """Create and save a visualization of topic similarity with proper formatting"""
    # Compute topic similarity matrix
    n_topics = lda_model.n_components
    topic_similarity = np.zeros((n_topics, n_topics))
    
    # Calculate all similarity values
    for i in range(n_topics):
        for j in range(n_topics):
            # Compute cosine similarity between topic vectors
            similarity = np.dot(lda_model.components_[i], lda_model.components_[j])
            similarity /= (np.linalg.norm(lda_model.components_[i]) * np.linalg.norm(lda_model.components_[j]))
            topic_similarity[i, j] = similarity
    
    # Explicitly set diagonal to 1.0 (topic self-similarity)
    np.fill_diagonal(topic_similarity, 1.0)
    
    # Create figure with adjusted size
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with explicit annotations and formatting
    ax = sns.heatmap(
        topic_similarity, 
        annot=True,          # Show the values
        fmt='.2f',           # Format to 2 decimal places
        cmap='YlGnBu',       # Color scheme
        cbar_kws={'label': 'Cosine Similarity'},
        xticklabels=[f'Topic {i+1}' for i in range(n_topics)],
        yticklabels=[f'Topic {i+1}' for i in range(n_topics)],
        annot_kws={"size": 8}  # Reduce annotation text size for better fit
    )
    
    # Adjust tick label size if needed
    plt.setp(ax.get_xticklabels(), fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    
    plt.title('Topic Similarity Matrix', fontsize=16)
    plt.tight_layout()
    
    # Save with higher DPI for better quality
    plt.savefig(os.path.join(output_dir, 'topic_similarity.png'), dpi=300, bbox_inches='tight')
    logging.info("Saved topic similarity visualization")

def visualize_document_clustering(doc_topic_dist, output_dir):
    """Create and save a visualization of document clustering by topic"""
    try:
        from sklearn.manifold import TSNE
        
        # Apply t-SNE to reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, doc_topic_dist.shape[0]-1))
        doc_tsne = tsne.fit_transform(doc_topic_dist)
        
        # Get dominant topic for each document
        dominant_topic = np.argmax(doc_topic_dist, axis=1) + 1
        
        # Create scatter plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(doc_tsne[:, 0], doc_tsne[:, 1], c=dominant_topic, cmap='tab20', alpha=0.4)
        plt.colorbar(scatter, label='Dominant Topic')
        plt.title('Document Clustering by Topic (t-SNE)', fontsize=16)
        plt.xlabel('t-SNE Dimension 1', fontsize=14)
        plt.ylabel('t-SNE Dimension 2', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'document_clustering.png'), dpi=200)
        logging.info("Saved document clustering visualization")
    except Exception as e:
        logging.error(f"Error creating document clustering visualization: {e}")

def save_results(lda_model, vectorizer, doc_topic_dist, metadata, output_dir, n_topics):
    """Save the model and results for future use"""
    # Save the LDA model and vectorizer
    with open(os.path.join(output_dir, 'lda_model.pkl'), 'wb') as f:
        pickle.dump(lda_model, f)
    
    with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Add topic distribution to metadata
    results_df = metadata.copy()
    
    # Add dominant topic
    results_df['dominant_topic'] = np.argmax(doc_topic_dist, axis=1) + 1
    
    # Add topic distribution columns
    for i in range(n_topics):
        results_df[f'topic_{i+1}_prob'] = doc_topic_dist[:, i]
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'topic_modeling_results.csv'), index=False)
    logging.info("Saved model and results to files")
    
    return results_df

def generate_topic_summary(lda_model, vectorizer, n_topics, n_top_words=15):
    """Generate a summary of topics with their top words"""
    feature_names = vectorizer.get_feature_names_out()
    topics_summary = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics_summary.append({
            'topic_id': topic_idx + 1,
            'top_words': top_features
        })
    
    return pd.DataFrame(topics_summary)

def create_html_summary(topic_summary, results_df, output_dir):
    """Create an HTML summary report with links to all visualizations"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Topic Modeling Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }
            h1, h2, h3 {
                color: #333;
            }
            .topic {
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                margin: 10px 0;
            }
            .viz-gallery {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                grid-gap: 15px;
                margin-top: 20px;
            }
            .viz-item {
                text-align: center;
            }
            .viz-item img {
                max-height: 200px;
                object-fit: contain;
            }
        </style>
    </head>
    <body>
        <h1>Topic Modeling Results</h1>
        
        <h2>Topic Summary</h2>
        <table>
            <tr>
                <th>Topic ID</th>
                <th>Top Words</th>
            </tr>
    """
    
    # Add topic summary rows
    for i, row in topic_summary.iterrows():
        html_content += f"""
            <tr>
                <td>{row['topic_id']}</td>
                <td>{', '.join(row['top_words'])}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Visualizations</h2>
        
        <h3>Topic Keywords</h3>
        <img src="topic_keywords.png" alt="Topic Keywords" />
        
        <h3>Document-Topic Distribution</h3>
        <img src="doc_topic_distribution.png" alt="Document-Topic Distribution" />
        
        <h3>Topic Similarity</h3>
        <img src="topic_similarity.png" alt="Topic Similarity" />
        
        <h3>Document Clustering</h3>
        <img src="document_clustering.png" alt="Document Clustering" />
        
        <h3>Topic Word Clouds</h3>
        <div class="viz-gallery">
    """
    
    # Add wordcloud images
    for i in range(1, topic_summary.shape[0] + 1):
        html_content += f"""
            <div class="viz-item">
                <img src="wordclouds/topic_{i}_wordcloud.png" alt="Topic {i} Word Cloud" />
                <p>Topic {i}</p>
            </div>
        """
    
    html_content += """
        </div>
        
        <h2>Results Summary</h2>
        <p>Total documents analyzed: """ + str(results_df.shape[0]) + """</p>
        <p>Number of topics: """ + str(topic_summary.shape[0]) + """</p>
        
    </body>
    </html>
    """
    
    # Write to file
    with open(os.path.join(output_dir, 'summary_report.html'), 'w') as f:
        f.write(html_content)
    
    logging.info("Created HTML summary report")

def analyze_directory(main_dir, output_dir='topic_model_results', n_topics=10):
    """
    Main function to analyze a directory of text files using topic modeling
    
    Args:
        main_dir (str): Path to the main directory containing subdirectories with txt files
        output_dir (str): Directory to save results and visualizations
        n_topics (int): Number of topics to extract
    """
    start_time = datetime.now()
    logging.info(f"Starting topic modeling analysis with {n_topics} topics")
    logging.info(f"Main directory: {main_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    # Setup NLTK
    # setup_nltk()
    
    # Read all text files
    texts, metadata = read_text_files(main_dir)
    
    if len(texts) == 0:
        logging.error("No text files found. Aborting.")
        return
    
    # Preprocess texts
    dtm, vectorizer = preprocess_texts(texts)
    
    # Run topic modeling
    lda_model, doc_topic_dist = run_topic_modeling(dtm, n_topics=n_topics)
    
    # Create visualizations
    create_visualizations(texts, metadata, lda_model, dtm, vectorizer, doc_topic_dist, output_dir, n_topics)
    
    # Save results
    results_df = save_results(lda_model, vectorizer, doc_topic_dist, metadata, output_dir, n_topics)
    
    # Generate topic summary
    topic_summary = generate_topic_summary(lda_model, vectorizer, n_topics)
    topic_summary.to_csv(os.path.join(output_dir, 'topic_summary.csv'), index=False)
    
    # Create HTML summary report
    create_html_summary(topic_summary, results_df, output_dir)
    
    # Print summary
    logging.info("\n----- Topic Modeling Summary -----")
    logging.info(f"Number of documents processed: {len(texts)}")
    logging.info(f"Number of topics extracted: {n_topics}")
    logging.info(f"Number of unique terms in vocabulary: {len(vectorizer.get_feature_names_out())}")
    logging.info("\nTop words for each topic:")
    
    for i, row in topic_summary.iterrows():
        logging.info(f"Topic {row['topic_id']}: {', '.join(row['top_words'][:10])}")
    
    logging.info(f"\nResults saved to {output_dir}")
    logging.info(f"Analysis completed in {datetime.now() - start_time}")
    
    return results_df, topic_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Topic modeling for a collection of text files')
    parser.add_argument('main_dir', type=str, help='Path to the main directory containing subdirectories with txt files')
    parser.add_argument('--output_dir', type=str, default='topic_model_results', help='Directory to save results and visualizations')
    parser.add_argument('--n_topics', type=int, default=10, help='Number of topics to extract')
    
    args = parser.parse_args()
    
    analyze_directory(args.main_dir, args.output_dir, args.n_topics)