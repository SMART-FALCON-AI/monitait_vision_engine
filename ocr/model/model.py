import easyocr
import pprint


class EasyOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['fa',], gpu=True, model_storage_directory="/code/models/")
        self.languages = ['fa',]

    def set_language(self, languages):
        self.languages = languages
        self.reader = easyocr.Reader(languages)

    def predict(self, image):
        results = list(self.reader.readtext(image, width_ths=1000, rotation_info=[0, 180], text_threshold=0.4, allowlist='ئچجحخهعغفقثصضگکمنتالبیسشظطزرذدپوژ', detail=0))
        print(results)
        # new_res = []
        # for res in results:
        #     box = res[0]
        #     text = res[1]
        #     # covnert box points to int
        #     for idx, point in enumerate(box):
        #         box[idx] = [int(p) for p in point]
        #     new_res.append({'box': box, 'text': text})
        new_res = "".join(results)
        return new_res

    def get_languages(self):
        return self.languages


def test():
    easy_ocr = EasyOCR()
    easy_ocr.predict('../tests/sample.jpg')


if __name__ == '__main__':
    test()

