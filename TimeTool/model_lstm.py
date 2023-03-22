from typing import Optional

from fastapi import FastAPI
import json
from pydantic import BaseModel
from functions import LSTM_diy
import uvicorn

from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Response
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd

app = FastAPI()
# class Item:
#     RMSE: int
#     real:
#     predict: Optional[str] = None
#

# @app.get("/")
# def read_root():

#     return "HELLO"


# @app.get("/files/")
# #def read_root(df_orig,split):
#
# def read_root(datapath,split) :


# @app.get("/items/")
# def read_items(datapath,split):
#     data=pd.read_csv(datapath,parse_dates=True, index_col=0,encoding='utf-8')
#     split=int(split)
#
#     rmse,output1,output2=LSTM_diy(data,split)
#     print({"RMSE": rmse,"real":output1,"predict":output2})
#
#     return {"RMSE": rmse,"real":output1,"predict":output2,}

from dataclasses import dataclass, field

from typing import List, Optional

from fastapi import FastAPI


#
# @dataclass
#
# class Item:
#
#     name: str
#
#     price: float
#
#     tags: List[str] = field(default_factory=list)
#
#     description: Optional[str] = None
#
#     tax: Optional[float] = None



app = FastAPI()




@app.get("/items/next")

def read_next_item(datapath,split):
    data = pd.read_csv(datapath, parse_dates=True, index_col=0, encoding='utf-8')
    split = int(split)

    rmse, output1, output2 = LSTM_diy(data, split)

    # return output1
    print({"RMSE" : rmse, "real" : output1, "predict" : output2})

    #
    # return {
    #     "RMSE": rmse,
    #     "real": [21143.18234083, 21132.14290085, 21117.92708966, 21113.03767511,21119.75613359, 21131.98203495, 21141.29642114],
    #     "predict":  [21143.18234083, 21132.14290085, 21117.92708966, 21113.03767511,21119.75613359, 21131.98203495, 21141.29642114],
    #
    # }


    return {
        "RMSE": rmse,
        "real": output1.tolist(),
        "predict": output2.tolist(),

    }






if __name__ == '__main__':
    uvicorn.run(app='model_lstm:app', host="127.0.0.1", port=8000, reload=True, debug=True)
