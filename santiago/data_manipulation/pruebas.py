'''função recursiva que receba um número inteiro e retorne um inteiro cujos algarismos são os
sucessores dos algarismos do número recebido'''
def sucessor(n):
    if n == 0:
        return 1
    else:
        return sucessor(n//10) * 10 + (n%10 + 1)
print(sucessor(1234))