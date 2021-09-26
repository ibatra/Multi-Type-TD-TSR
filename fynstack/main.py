import os

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

# import scripts.TSR.table_detection as table_detection
# import scripts.table_ocr as tocr
# import scripts.TSR.table_structure_recognition_all as tsra
# import scripts.TSR.table_structure_recognition_wol as tsrwol
# import scripts.TSR.table_structure_recognition_lines_wol as tsrlwol
import scripts.TSR.table_structure_recognition_lines as tsrl
from utils import make_prediction, convert_boxes_2_dataframe, convert_pdf_2_img

# setup the API key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'fynstack-a5916aa2d765.json'

logger = setup_logger()
logger.info(f"Cuda available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file('detectron_weights/All_X152.yaml')
    cfg.MODEL.WEIGHTS = 'detectron_weights/model_final.pth'  # Set path model .pth
    predictor = DefaultPredictor(cfg)

    convert_pdf_2_img("pdfs/Financial_Results_Q3FY21.pdf", img_filename='test_img')

    logger.info("Detecting tables")
    table_list, table_coords = make_prediction("pdfs/test_img_1.jpg", predictor)
    table_list = [table_list[0]]

    logger.info("Recognizing table structure")
    finalboxes, output_img = tsrl.recognize_structure(table_list[0])
    df = convert_boxes_2_dataframe(finalboxes, table_list[0])
