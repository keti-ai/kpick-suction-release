import socket, threading
import numpy as np
import time
from ..import_basic_utils import *
from .base import BasObj

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def send_result(s, inf):
    s.send(inf.encode())

def byte2dict(byteData):
    byte_dict = eval(byteData.decode())
    data_dict = dict()
    for key in byte_dict:
        item = byte_dict[key]
        data = item['data']
        if item['is_array']: data = np.fromstring(data, dtype=item['dtype']).reshape(item['shape'])
        data_dict.update({key: data})
    return data_dict

def dict2byte(data_dict):
    out_dict = dict()
    for key in data_dict:
        value = data_dict[key]
        is_array = isinstance(value, np.ndarray)
        shape, dtype = None, None
        if is_array:
            shape, dtype = value.shape, value.dtype.name
            value = value.tostring()
        out_dict.update({key: {'is_array': is_array, 'shape': shape, 'dtype':dtype, 'data':value}})
    return str(out_dict).encode()

def ndarray2byteDict(mat):
    return {'size': mat.shape, 'dtype': mat.dtype.name, 'bytes': mat.tostring()}

def byteDict2mat(byteDict):
    return np.fromstring(byteDict['bytes'], dtype=byteDict['dtype']).reshape(byteDict['size'])


class ServerThread():
    def __init__(self, host='localhost', port=8888):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))

    def listen(self):
        print('{} listening a connection'.format('+'*10))
        self.sock.listen(1)
        while True:
            self.client, self.address = self.sock.accept()
            self.client.settimeout(1200)        # time out after 20 mins
            threading.Thread(target = self.listenToClient).start()

    def listenToClient(self):
        while True:
            try:
                print('{} waiting data'.format('+'*10))
                self.length = recvall(self.client, 16)

                if self.length is None:
                    raise ('Client disconnected')

                else:
                    print('{} Connected to {}'.format('+'*10,self.address))

                    byteData = self.receive_data()
                    self.data = byte2dict(byteData=byteData)
                    print('{}  Data received'.format('+'*10))


                    rets = self.process_received_data()
                    print('{}  Data processed'.format('+'*10))

                    self.send_return(rets)
                    print('{}  Return sent'.format('+'*10))
                    time.sleep(2)
                    ProcUtils().clscr()
            except:
                self.client.close()
                return False

    def receive_data(self):
        byteData =  recvall(self.client, int(self.length))
        return byteData
        # return self.get_data(byteData=byteData)

    def process_received_data(self):
        return self.data

    def send_return(self, rets):
        byteRets= dict2byte(data_dict=rets)
        byteLen = str(len(byteRets)).rjust(16, '0').encode()
        self.client.sendall(byteLen + byteRets)


class ClientThread():
    def __init__(self,  host='localhost', port=8888):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.host = host
        self.port = port
        self.server_connected = False

    def send_and_get_return(self, aDict):
        """ send a dict to sever"""
        while True:
            try:
                if not self.server_connected:
                    print('{} Connecting to server: {}'.format('+'*10,self.host))
                    self.sock.connect((self.host, self.port))
                    print('Connected ....')
                    self.server_connected = True

                self.send_dict(aDict=aDict)
                return self.get_return()
            except:
                self.sock.close()
                print('socket closed ...')
                return None

    def send_dict(self, aDict):
        print('{} Sending byte data'.format('+' * 10))
        byteData = dict2byte(data_dict=aDict)

        len_byte = str(len(byteData)).rjust(16, '0').encode()
        self.sock.send(len_byte + byteData)
        print('Byte data sent...')

    def get_return(self):
        print('{} Waiting for return'.format('+' * 10))
        ret_len = recvall(self.sock, 16)
        byteData = recvall(self.sock, int(ret_len))
        print('Return received ...')
        return byte2dict(byteData=byteData)




class DectectTcpServer():
    def __init__(self, args=None, cfg_path=None, detector=None, detector_reload=None):
        if cfg_path is not None: args = CFG(cfg_path=cfg_path)
        self.args = args
        self.detector = detector
        self.detector_reload = detector_reload


    def predict(self, inputs):
        return self.detector(inputs)

    def run(self):
        server = ServerThread(host='', port=self.args.port, detector=self.predict, detector_reload=self.detector_reload)
        server.listen()

class DetectTcpClient():

    def predict(self, inputs, filename='unnamed'):
        for name in inputs:
            if not isinstance(inputs[name], np.ndarray):continue
            inputs[name] = ndarray2bytes(inputs[name])
        inputs.update({'args': str(self.args.todict())})
        byteData = str(inputs).encode()
        byteRets = ClientThread(self.args.host, self.args.port).sendMessage(byteData)
        # rets = pickle.loads(byteRets)
        if byteRets is None: return None
        rets = eval(byteRets.decode())
        if not (rets, (list, tuple)): rets = [rets]
        for i in range(len(rets)):
            ret = rets[i]
            for name in ret:
                if not isinstance(ret[name], bytes): continue
                rets[i][name] = bytes2ndarray(ret[name])
        return rets

    def run(self):
        server = ServerThread(host='', port=self.args['port'], detector=self.predict)
        server.listen()



def ndarray2bytes(mat):
    info = mat.dtype.name
    for s in mat.shape:
        info+= '_' + str(s)
    info = info.ljust(32, '$')
    return info.encode() + mat.tostring()

def bytes2ndarray(byteData):
    info = byteData[:32].decode()
    info = info[:info.find('$')]
    info = tuple(el for el in info.split('_'))
    shape = tuple(int(el) for el in info[1:])
    data_type = info[0]
    return np.fromstring(byteData[32:], dtype=data_type).reshape(shape)

if __name__=='__main__':
    pass











