"""
class:
    TokenMapper(self, text)
"""

class TokenMapper:
    """
        Mapping arbitrary tokens to integer range(0, N)
    """
    def __init__(self, tokens=[]):
        self.__encoder = {}
        self.__decoder = {}
        self.__id = 0
        self.append(tokens)

    def __len__(self):
        return len(self.__encoder) + 1    # extra one token for <unknown>

    def append(self, tokens):
        for token in set(tokens):
            if token not in self.__encoder:
                self.__encoder[token] = self.__id
                self.__decoder[self.__id] = token
                self.__id += 1

    def encode(self, tokens):
        # return a special value for unknown token which can be decoded as 'unknown'
        encoded = [self.__encoder.get(token, len(self.__encoder)) for token in tokens]
        return encoded

    def decode(self, encoded, unknown=None):
        # return value 'unknown' for unknown token
        decoded = [self.__decoder.get(encoded_token, unknown) for encoded_token in encoded]
        return decoded
    
if __name__ == "__main__":
    tokens = [1199, 4791, 14064, 4211, 3085, 31069, 11542, 311, 8641, 9886, 1555, 5933, 4221, 13, 14968, 11, 433, 374, 25420, 1268, 1778, 11542, 649, 387, 62113, 311, 7068, 5448, 315, 3230, 5016, 19476, 11, 5719, 872, 11341, 11, 477, 31435, 1124, 304, 502, 13073, 323, 11775, 16451, 13, 763, 1023, 4339, 11, 584, 2610, 25, 1268, 649, 584, 1005, 4221, 84792, 291, 4211, 311, 2543, 1057, 8415, 1139, 264, 19354, 11, 477, 13085, 264, 502, 2027, 3196, 389, 1057, 7075, 22068, 30, 5810, 584, 3118, 264, 4382, 5603, 430, 6276, 1778, 11782, 11542, 13, 12362, 1193, 220, 18, 12, 20, 5448, 315, 264, 1217, 10039, 44057, 7434, 11, 1093, 459, 1665, 477, 264, 1742, 11, 584, 4048, 311, 4097, 433, 1555, 502, 330, 5880, 1, 304, 279, 40188, 3634, 315, 264, 20268, 1495, 4791, 14064, 1646, 13, 4314, 330, 5880, 1, 649, 387, 24306, 1139, 5933, 4221, 23719, 11, 51346, 35649, 9886, 304, 459, 42779, 1648, 13, 2876, 2915, 11, 584, 1505, 6029, 430, 264, 3254, 3492, 40188, 374, 14343, 369, 40880, 5016, 323, 28830, 19476, 13, 1226, 9616, 1057, 5603, 311, 264, 7029, 2134, 315, 3122, 11243, 11, 323, 20461, 430, 433, 649, 810, 94176, 25920, 279, 19476, 4028, 264, 2134, 315, 8522, 323, 9256, 627, 8140, 2082, 11, 828, 323, 502, 4339, 690, 387, 2561, 520, 25, 420, 3788, 5665]
    mapper = TokenMapper(tokens)

    enc_tokens = mapper.encode(tokens)
    dec_tokens = mapper.decode(enc_tokens)

    print(set(enc_tokens), len(mapper))
    print(dec_tokens == tokens)

    tokens_with_unknown = tokens[:5] + [2233]
    print(mapper.decode(mapper.encode(tokens_with_unknown)))