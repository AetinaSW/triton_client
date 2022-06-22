import json, os

class JsonParser:

    def __init__(self, path) -> None:
        
        self.path = path
        self.data = self.load_data()

    """ 回傳 JSON 檔案的內容，如果沒給就會吃 self.path """
    def load_data(self, path:str="") -> dict:

        trg_path = path if path != "" else self.path

        if not os.path.exists(trg_path):
            raise Exception('File is not exists !')
        elif os.path.splitext(trg_path)[1] != '.json':
            raise Exception("It's not a json file ({})".format(trg_path))
        else:
            with open(trg_path) as file:
                data = json.load(file)  # load is convert dict from json "file"
            
        return data

    """ 回傳 JSON 內容 """
    def get_data(self):
        return self.data

    """ 寫入 JSON 檔案，如果沒給 path 就會吃 self.path """
    def write_data(self, new_cnt:dict, path:str="") -> None:
        
        trg_path = path if path != "" else self.path
        
        with open(trg_path, 'w') as file:
            json.dump(new_cnt, file)    # dump is write dict into file

if __name__ == '__main__':
    
    path = './configs/models.json'
    print(JsonParser(path).get_data())