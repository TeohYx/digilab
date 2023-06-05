import streamlit as st
import generate as g
from sentences import Sentences
import time
import file
import gdown

# # link to model
# url = 'https://drive.google.com/file/d/1fTUGsXiIz_egy4qM_R4Jujg2gCNFuZsX/view?usp=sharing'
# output = 'atomic-thunder-15-7.dat'
# gdown.download(url, output, quiet=False)

st.set_page_config(page_title="Dialect Translator", page_icon=":tada:")

spinner = None
with st.container():
    st.write("---")
    st.header("English to Cantonese Translator")
    st.write("##")

    input_data = st.text_input('English Sentences:', value="")
    submit_button = st.button('Generate')
    audio_button = st.button('Record Audio')

    if audio_button:
        file.record_audio()
        if Sentences.get_sentence() != "":   
            st.write("#")
            st.write("Translated Cantonese:")
            with st.spinner():
                g.generate(Sentences.get_sentence())
                while Sentences.is_translated() is False:
                    st.write('Translating...')
            # input_data.write(Sentences.get_sentence())
            # st.text_input('English Sentences:', value=Sentences.get_sentence())      
            print("input data: " + input_data)
        else:
            # st.write(Sentences.get_popupmsg())
            st.error(Sentences.get_popupmsg())

    if submit_button:
        # Call the function in other_file.py and pass the input_data as an argument
        st.write("#")
        st.write("Translated Cantonese:")
        with st.spinner():
            g.generate(input_data)
            while Sentences.is_translated() is False:
                st.write('Translating...')

    if Sentences.is_translated() is True:
        st.write(Sentences.get_cantonese())

    # contact_form = """
    # <form action="/generate" method="POST">
    #     <input type="hidden" name="_captcha" value="false">
    #     <input type="text" id="filename" name="filename">
    #     <button type="submit">Generate</button>
    # </form>
    # """

    # left_column, right_column = st.columns(2)
    # with left_column:
    #     st.markdown(contact_form, unsafe_allow_html=True)
    # with right_column:
    #     st.empty()