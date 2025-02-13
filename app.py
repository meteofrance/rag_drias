import os
import time

import streamlit as st
from rzg_drias.settings import PATH_FEEDBACK

from rag_drias.app_utils import add_json_with_lock

# Add IS_STREAMLIT to the environment
os.environ["IS_STREAMLIT"] = "True"
from main import answer  # noqa: E402

correct_password = st.secrets["general"]["password"]


if "password_valid" not in st.session_state:
    st.session_state.password_valid = False

# Password protection
if not st.session_state.password_valid:
    # ------------ Password protection page -------
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if username == "" or password == "":
        st.stop()
    elif password == correct_password:
        st.session_state.username = username
        st.session_state.password_valid = True
        st.success("Correct password")
        # Reload the app to show the main interface
        st.rerun()
    else:
        st.error("Incorrect password")
        st.stop()
else:
    # ------------ Main interface ------------

    # Header
    col1, _, col2 = st.columns([2, 4, 2])
    with col1:
        st.image(
            "https://www.drias-climat.fr/public/images/logo_final+Etat.jpg", width=300
        )
    with col2:
        st.image("https://www.drias-eau.fr/public/images/Logo_DRIAS.jpg", width=125)

    st.title("💬☀️ Chatbot DRIAS")

    st.write(
        "Bienvenue sur le chatbot DRIAS, un assistant virtuel qui vous aidera à trouver des informations en se basant\
         sur les données du site [DRIAS](https://www.drias-climat.fr/).\n\nLorsque l'option *use rag* est activée, le\
         chatbot va parcourir l'ensemble des textes présents sur le site et identifie un nombre *Number of retrieved\
         chunks* de paragraphes qui ont l'air pertinents pour répondre à la question. Puis une instruction sera donnée\
         au *generative model*: \"voici des documents : [paragraphe 1], [paragraphe 2], etc. A partir de ces documents\
        , répond à la question : [question utilisateur]\".\n\nPour commencer, sélectionnez les paramètres de votre\
         choix dans la barre latérale puis posez votre question dans la zone de texte ci-dessous.\
        \n\nN'hésitez pas à donner votre avis sur le chatbot en cliquant sur ce \
        [lien](https://docs.google.com/forms/d/1pT3kqqPp6OiV0XPY7cSPkznasIbgW8tL3UctR-Ox2Kk/edit)."
    )

    # Sidebar with parameters
    st.sidebar.title("Parameters")

    use_rag = st.sidebar.checkbox(
        "Use rag",
        value=True,
        help="Utiliser le RAG (Retrieval augmented generation) permet de générer des réponses plus précises en se\
             basant sur des morceaux de documents récupérés.\nSi cette option est désactivée, le chatbot générera des\
             réponses sans se baser sur le site DRIAS.",
    )

    generative_model = st.sidebar.selectbox(
        "Choose a generative model:",
        [
            "Llama-3.2-3B-Instruct",
            "Chocolatine-3B-Instruct-DPO-v1.0",
            "DeepSeek-R1-Distill-Llama-8B",
        ],
        help="Modèle de génération de texte utilisé pour répondre aux questions. \nLLama-3.2-3B-Instruct\
             est recommandé.",
    )

    alpha = st.sidebar.slider(
        "Hybride search alpha :",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        disabled=not use_rag,
        help="Paramètre de pondération entre la recherche semantique (vectorielle) et la recherche lexicale (BM25).\
             \nUn alpha proche de 0 donnera plus de poids à la recherche lexicale et un alpha proche de 1 donnera plus\
             de poids à la recherche semantique.",
    )

    n_samples = st.sidebar.slider(
        "Number of retrieved chunks :",
        min_value=5,
        max_value=100,
        value=40,
        disabled=not use_rag,
        help="Nombre de morceaux de documents provenant du site DRIAS récupérés pour chaque question.\nPlus le nombre\
             est grand, plus le chatbot aura de contexte mais plus le temps de calcul sera long.\nLorsque le nombre de\
             morceaux est élevé, il est recommandé d'utiliser un modèle de reranking.",
    )

    reranker_model = st.sidebar.selectbox(
        "Choose a reranker model:",
        ["bge-reranker-v2-m3", "No reranker"],
        disabled=not use_rag,
        help="Modèle de reranking utilisé selctionner les morceaux de documents les plus important parmis ceux\
             recupérés et les classer par ordre de pertinence. \nUtiliser un modèle de reranking permet d'améliorer\
             la qualité des réponses mais augmente le temps de calcul.\nIl est recommandé d'utiliser un modèle de\
             reranking lorsque le nombre de morceaux de documents récupérés est élevé.",
    )

    if reranker_model == "No reranker":
        reranker_model = ""

    use_pdf = st.sidebar.checkbox(
        "PDFs in database",
        value=False,
        disabled=not use_rag,
        help="Si cette option est activée, les PDFs qui sont en lien sur le portail DRIAS seront dans la base\
             de données utilisée pour le RAG. Cela permet d'avoir plus d'informations mais celles ci ne sont pas\
             toujours pertinentes.",
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Comment puis-je vous aider ?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Streamed response emulator
        def response_generator():
            response = answer(
                prompt,
                generative_model=generative_model,
                n_samples=n_samples,
                use_rag=use_rag,
                reranker=reranker_model,
                use_pdf=use_pdf,
                alpha=alpha,
            )

            for word in response.split(" "):
                yield word + " "
                time.sleep(0.02)

        def save_feedback(index: int):
            """Save feedback in a json file"""
            dict_params = {
                "use_pdf": use_pdf,
                "use_rag": use_rag,
                "generative_model": generative_model,
                "alpha": alpha,
                "n_samples": n_samples,
                "reranker_model": reranker_model,
            }
            add_json_with_lock(
                PATH_FEEDBACK,
                {
                    "username": st.session_state.username,
                    "query": prompt,
                    "response": response,
                    "feedback": st.session_state[f"feedback_{index}"],
                    "params": dict_params,
                },
            )

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator())
            # Display stars for feedback
            selected = st.feedback(
                "stars",
                key=f"feedback_{len(st.session_state.messages)}",
                on_change=save_feedback,
                args=[len(st.session_state.messages)],
            )

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
