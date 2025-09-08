import sys
import traceback
import logging
from Churn_Pred.logger.log import logging


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = CustomException.get_detailed_error_message(
            error_message, error_detail
        )

    @staticmethod
    def get_detailed_error_message(error: Exception, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"""
            [CustomException]
            Error Type     : {type(error).__name__}
            File           : {file_name}
            Line Number    : {line_number}
            Error Message  : {error}
            """.strip()
        else:
            return f"[CustomException] {type(error).__name__}: {error}"

    def __str__(self):
        return self.error_message


if __name__=='__main__':
    try:
        1 / 0
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(custom_error)
        raise custom_error