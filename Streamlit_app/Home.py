import os
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import streamlit as st
import math
import re

st.markdown("""
<style>
            
            .css-cio0dv.ea3mdgi1
            {
            visibility:hidden;
            }
            .css-10trblm.e1nzilvr0
            {
            display: flex;
            justify-content: center;
            align-items: center;
            }
            </style>
""", unsafe_allow_html = True)

# Function to preprocess the text data
def preprocess_text(text):
    if isinstance(text, str):  # Check if it's a non-null string
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text
        words = nltk.word_tokenize(text)
        # Remove punctuation and non-alphabetic characters
        words = [word for word in words if word.isalpha()]

        # Remove special characters
        words = [re.sub(r'[^A-Za-z0-9\s]', '', word) for word in words]
        
        # Remove numbers
        words = [word for word in words if not word.isdigit()]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)
    else:
        return ''  # Return an empty string for NaN values


# Function to generate the word cloud
def generate_wordcloud(top_words_text):
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(top_words_text)
    #plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    #plt.title('Word Cloud')
    plt.axis("off")
    st.pyplot(plt)

@st.cache_data
# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, file_path)
    df = pd.read_csv(file_path)
    #df = df.head(1000)
    df['Corpus'] = df['Title'] + ' ' + df['Abstract'] + ' ' + df['Category']
    df['Processed_Corpus'] = df['Corpus'].apply(preprocess_text)
    return df

@st.cache_data
# Function to build the LDA model
def build_lda_model(processed_corpus, num_topics=10):
    tokenized_corpus = processed_corpus.apply(nltk.word_tokenize)
    dictionary = Dictionary(tokenized_corpus)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_corpus]
    lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model


# Main function for Streamlit app
def main_page():
    # Load and preprocess data
    df = load_and_preprocess_data("ensemble_results.csv")

    # Build the LDA model
    num_topics = 10
    lda_model = build_lda_model(df['Processed_Corpus'], num_topics=num_topics)

    # Merge the top words from each topic into a single text
    top_words_text = ''
    for topic_num in range(num_topics):
        topic_words = lda_model.show_topic(topic_num, topn=4)
        top_words = [word for word, _ in topic_words if len(word) > 3 and not word.endswith("ing")]
        top_words_text += ' '.join(top_words) + ' '

    # Create Streamlit app
    st.title('IBiDAV')          

    # Search bar and button
    search_query = st.text_input('Enter your search query:', placeholder='Type PMID / PMCID / Title')
    search_button = st.button('Search')

    st.write("---")
    if search_button or search_query:
        if search_query.strip() == "":
            st.warning("Please enter query")
        else:
            if search_button or search_query:
                # Split the search query into individual words
                words = search_query.split('+')
                
                # Create a regular expression pattern to match any of the words
                pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word.strip()) for word in words) + r')\b', flags=re.IGNORECASE)

                # Define a lambda function to perform the matching
                pmid_match = df['PMID'].astype(str).apply(lambda x: bool(pattern.search(x)))
                pmcid_match = df['PMCID'].astype(str).apply(lambda x: bool(pattern.search(x)))
                title_match = df['Title'].apply(lambda x: bool(pattern.search(x)))

                # Convert non-string 'Abstract' values to strings
                df['Abstract'] = df['Abstract'].apply(lambda x: str(x))
                abstract_match = df['Abstract'].apply(lambda x: bool(pattern.search(x)))
                # Filter data based on the search query
                filtered_df = df[pmid_match | 
                                pmcid_match | 
                                title_match |
                                abstract_match
                                ]

                # Drop duplicates based on the columns you need
                columns_to_check_duplicates = ['PMCID', 'PMID', 'Title', 'Abstract', 'Article URL']
                filtered_df = filtered_df.drop_duplicates(subset=columns_to_check_duplicates)

                # Display the filtered DataFrame
                st.dataframe(filtered_df[['PMCID','PMID','Title','Abstract','Article URL']],use_container_width = True,hide_index = True)

                # Clear the word cloud placeholder to remove it from the output
                st.empty()            

    # Only generate the word cloud if the search button is not clicked
    if not (search_button or search_query):
        generate_wordcloud(top_words_text)

def display_filtered_data(category:str):
    df = load_and_preprocess_data("ensemble_results.csv")

    # Convert non-string 'Abstract' values to strings
    df['Abstract'] = df['Abstract'].apply(lambda x: str(x))
    # Filter data based on the search query
    filtered_df = df[(df['Category'] == category) & (df['multi_labels'] == category)]
    filtered_df = filtered_df.groupby(['PMCID', 'PMID', 'Title', 'Abstract', 'Article URL']).agg({
        'Image URL': lambda x: ', '.join(x)
        }).reset_index()
    # Rename columns if needed
    filtered_df.rename(columns={'Image URL': 'Image URLs'}, inplace=True)

    # Drop duplicates based on the columns you need
    columns_to_check_duplicates = ['PMCID', 'PMID', 'Title', 'Abstract', 'Article URL']
    filtered_df = filtered_df.drop_duplicates(subset=columns_to_check_duplicates)
    st.title(category)
    # Create a text input for search
    search_query = st.text_input('Enter your search query:', placeholder='Type PMID / PMCID / Title')
    search_button = st.button("Search")

    # Reset the applied search query and displayed papers count if search query is empty
    if search_query == "":
        if hasattr(st.session_state, 'applied_search_query'):
            del st.session_state.applied_search_query

    # Apply the search query filtering if applicable
    if search_button or search_query:
        st.session_state.applied_search_query = search_query

    # Apply the stored search query
    if hasattr(st.session_state, 'applied_search_query') and st.session_state.applied_search_query:
        search_query = st.session_state.applied_search_query
        # Split the search query into individual words
        words = search_query.split('+')
        
        # Create a regular expression pattern to match any of the words
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word.strip()) for word in words) + r')\b', flags=re.IGNORECASE)

        # Define a lambda function to perform the matching
        pmid_match = filtered_df['PMID'].astype(str).apply(lambda x: bool(pattern.search(x)))
        pmcid_match = filtered_df['PMCID'].astype(str).apply(lambda x: bool(pattern.search(x)))
        title_match = filtered_df['Title'].apply(lambda x: bool(pattern.search(x)))
        abstract_match = filtered_df['Abstract'].apply(lambda x: bool(pattern.search(x)))

        filtered_df = filtered_df[pmid_match |
                pmcid_match |
                title_match |
                abstract_match
            ]
        
        if filtered_df.empty:
            pass
            
        
    pmids = list(filtered_df['PMID'])
    pmcids = list(filtered_df['PMCID'])
    titles = list(filtered_df['Title'])
    abstracts = list(filtered_df['Abstract'])
    urls = list(filtered_df['Article URL'])
    iurls = list(filtered_df['Image URLs'])

    num_initial_papers = 6
    num_papers_per_load = 3

    # Get the current number of displayed papers from the session state
    if 'num_displayed_papers' not in st.session_state:
        st.session_state.num_displayed_papers = num_initial_papers

    
    # Create a container to hold the card layout
    container = st.container()

    # Iterate through papers
    for i in range(st.session_state.num_displayed_papers):
        if i < len(pmids) and i < len(pmcids):
            with container:
                # Create a card container for consistent sizing
                card = st.container()

                # Place the content inside the card
                with card:
                    # Use columns to structure the layout
                    col1, col2 = st.columns([6, 1])  # Adjust the ratios as needed
                    
                    # Column 1: Display all content except images
                    with col1:
                        with st.container():
                            st.write("---")
                            st.write(f"PMID: {pmids[i]} , PMCID: {pmcids[i]}") 
                            st.write(f"Title: {titles[i]}")
                            
                            # Handle NaN values in the abstract
                            abstract = abstracts[i]
                            if isinstance(abstract, str):
                                abstract = abstract[:100] + "..." if len(abstract) > 100 else abstract 
                                st.write(f"Abstract: {abstract[:100]}...")
                            else:
                                abstract = ""
                                st.write(f"Abstract: Not Available")
                            st.write(f"Article URL: {urls[i]}")
                            #st.write(f"Image URL: {iurls[i]}")
                    
                    # Column 2: Display images as thumbnails
                    with col2:
                        with st.container():
                            st.write("---")
                            # Split the comma-separated URLs
                            image_urls = iurls[i].split(',')
                            
                            # Calculate the number of rows and columns for the grid
                            num_images = min(len(image_urls),6)
                            num_columns = 3  # Adjust the number of columns as needed
                            num_rows = int(math.ceil(num_images / num_columns))
                            
                            # Create a grid of image thumbnails
                            for row in range(num_rows):
                                with st.container():
                                    for col in range(num_columns):
                                        image_index = row * num_columns + col
                                        if image_index < num_images:
                                            st.image(image_urls[image_index].strip(), width=50)  # Adjust width for thumbnails
                            #st.write("---")

                # Apply a consistent height style to the card
                card.markdown(
                    f"""<style>
                        .reportview-container .main .block-container {{
                            width: 100%;  
                            height: auto;
                        }}
                    </style>""",
                    unsafe_allow_html=True
                )
    st.markdown(
        f"Displaying {st.session_state.num_displayed_papers} out of {len(pmids)} results.",
        unsafe_allow_html=True
    )

    # Check if there are more papers to load
    if st.session_state.num_displayed_papers < len(pmids):
        if st.button("Load More"):
            # Increment the number of displayed papers
            st.session_state.num_displayed_papers += num_papers_per_load

            

    # Display a message if all papers have been loaded
    else:
        st.write("All papers have been loaded.")

    pass

def page2():
    # page for Computed Tomography (CT) Scan
    display_filtered_data("Computed Tomography (CT) Scan")
    pass

def page3():
    # page for Endoscopy
    display_filtered_data("Endoscopy")
    pass

def page4():
    # page for Histology
    display_filtered_data("Histology")    
    pass

def page5():
    # page for Magnetic Resonance Imaging (MRI)
    display_filtered_data("Magnetic Resonance Imaging (MRI)")    
    pass

def page6():
    # page for Positron Emission Tomography (PET) Scan
    display_filtered_data("Positron Emission Tomography (PET) Scan") 
    pass

def page7():
    # page for Ultrasound
    display_filtered_data("Ultrasound")
    pass

def page8():
    # page for X-ray
    display_filtered_data("X-ray")
    pass

def page9():
    st.title('IBiDAV')
    st.write("""
            Welcome to IBiDAV : Integrative Biomedical Data Analysis and Visualization.
             
             An ongoing master thesis project designed to revolutionize the way we explore and analyze biomedical data. 
            IBiDAV is a dynamic platform that seamlessly integrates cutting-edge image classification and natural language processing. 
             
            Dive into a vast repository of medical images and scholarly literature, categorize papers, uncover hidden themes, and conduct intricate searches. 
            Our intuitive interface serves as a bridge between image and text analysis, providing a comprehensive understanding of the biomedical domain.
             
            Join us on this transformative journey through the intricate landscape of biomedical research. 
            Let IBiDAV empower you to discover, learn, and innovate in the world of healthcare and life sciences.             
             """)
    st.write("""
    Developed by - Piyush Kumar Singh

    Guided by - Prof. Soldatos, Theodoros
    """)
    pass

page_names_to_funcs = {
    "Home": main_page,
    "Computed Tomography (CT) Scan": page2,
    "Endoscopy": page3,
    "Histology": page4,
    "Magnetic Resonance Imaging (MRI)": page5,
    "Positron Emission Tomography (PET) Scan": page6,
    "Ultrasound": page7,
    "X-ray": page8,
    "About": page9
}

selected_page = "Home"

# Create session state for storing selected page
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Home"

# Create clickable buttons in the sidebar
for page_name in page_names_to_funcs.keys():
    if st.sidebar.button(page_name, use_container_width=True):
        # Reset the applied search query and displayed papers count when switching pages
        if hasattr(st.session_state, 'applied_search_query'):
            del st.session_state.applied_search_query
        st.session_state.selected_page = page_name
        st.session_state.num_displayed_papers = 6

# Call the selected page's function
page_names_to_funcs[st.session_state.selected_page]()

# This block runs only when the script is executed as the main program
if __name__ == "__main__":
    # Streamlit app setup and execution
    pass