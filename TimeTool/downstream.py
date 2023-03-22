import pathlib
import pandas as pd
import streamlit as st

# HACK This only works when we've installed streamlit with pipenv, so the
# permissions during install are the same as the running process
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
# We create a downloads directory within the streamlit static asset directory
# and we write output files to it
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

def writecsv(mydataframe,name):


    #mydataframe = pd.DataFrame.from_dict({'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']})
    mydataframe.to_csv(str(DOWNLOADS_PATH / name), index=True)
    #st.markdown("Download from [downloads/output.csv](downloads/output.csv)")
    return DOWNLOADS_PATH



