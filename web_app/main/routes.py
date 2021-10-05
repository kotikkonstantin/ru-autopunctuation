# -*- coding: utf-8 -*-
import http
from flask import Response, current_app, request

from inference import inference
from web_app.main import bp


@bp.route('/healthcheck')
def healthcheck():
    return Response('OK', status=http.HTTPStatus.OK)


@bp.route('/predict', methods=["POST"])
def predict():

    try:
        input_data = request.get_json()
        input_data = input_data['corpus']
        output_list = []
        for text in input_data:
            punc_case_restored_text = inference(text, current_app.model,
                                                current_app.tokenizer,
                                                current_app.device,
                                                current_app.model_hyperparameters,
                                                current_app.config.get("BATCH_SIZE"))
            output_list.append(punc_case_restored_text)

        result = {"output": output_list}
        return result
    except KeyError as e:
        return Response(str(e), status=http.HTTPStatus.BAD_REQUEST)

