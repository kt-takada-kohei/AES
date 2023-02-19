# AES-128
# 平文の上限を128bitに制限したver
# それ以降はもう一度実行

import sys              # 実行環境に関するライブラリ
import random           # 疑似乱数生成ライブラリ
import numpy as np      # 多次元配列などを扱うライブラリ
import sympy            # 代数演算ライブラリ

# S-boxの定義
sbox = {'00000000':'01100011', '00000001':'01111100', '00000010':'01110111', '00000011':'01111011', '00000100':'11110010', '00000101':'01101011',
        '00000110':'01101111', '00000111':'11000101', '00001000':'00110000', '00001001':'00000001', '00001010':'01100111', '00001011':'00101011',
        '00001100':'11111110', '00001101':'11010111', '00001110':'10101011', '00001111':'01110110', '00010000':'11001010', '00010001':'10000010',
        '00010010':'11001001', '00010011':'01111101', '00010100':'11111010', '00010101':'01011001', '00010110':'01000111', '00010111':'11110000',
        '00011000':'10101101', '00011001':'11010100', '00011010':'10100010', '00011011':'10101111', '00011100':'10011100', '00011101':'10100100',
        '00011110':'01110010', '00011111':'11000000', '00100000':'10110111', '00100001':'11111101', '00100010':'10010011', '00100011':'00100110',
        '00100100':'00110110', '00100101':'00111111', '00100110':'11110111', '00100111':'11001100', '00101000':'00110100', '00101001':'10100101',
        '00101010':'11100101', '00101011':'11110001', '00101100':'01110001', '00101101':'11011000', '00101110':'00110001', '00101111':'00010101',
        '00110000':'00000100', '00110001':'11000111', '00110010':'00100011', '00110011':'11000011', '00110100':'00011000', '00110101':'10010110',
        '00110110':'00000101', '00110111':'10011010', '00111000':'00000111', '00111001':'00010010', '00111010':'10000000', '00111011':'11100010',
        '00111100':'11101011', '00111101':'00100111', '00111110':'10110010', '00111111':'01110101', '01000000':'00001001', '01000001':'10000011',
        '01000010':'00101100', '01000011':'00011010', '01000100':'00011011', '01000101':'01101110', '01000110':'01011010', '01000111':'10100000',
        '01001000':'01010010', '01001001':'00111011', '01001010':'11010110', '01001011':'10110011', '01001100':'00101001', '01001101':'11100011',
        '01001110':'00101111', '01001111':'10000100', '01010000':'01010011', '01010001':'11010001', '01010010':'00000000', '01010011':'11101101',
        '01010100':'00100000', '01010101':'11111100', '01010110':'10110001', '01010111':'01011011', '01011000':'01101010', '01011001':'11001011',
        '01011010':'10111110', '01011011':'00111001', '01011100':'01001010', '01011101':'01001100', '01011110':'01011000', '01011111':'11001111',
        '01100000':'11010000', '01100001':'11101111', '01100010':'10101010', '01100011':'11111011', '01100100':'01000011', '01100101':'01001101',
        '01100110':'00110011', '01100111':'10000101', '01101000':'01000101', '01101001':'11111001', '01101010':'00000010', '01101011':'01111111',
        '01101100':'01010000', '01101101':'00111100', '01101110':'10011111', '01101111':'10101000', '01110000':'01010001', '01110001':'10100011',
        '01110010':'01000000', '01110011':'10001111', '01110100':'10010010', '01110101':'10011101', '01110110':'00111000', '01110111':'11110101',
        '01111000':'10111100', '01111001':'10110110', '01111010':'11011010', '01111011':'00100001', '01111100':'00010000', '01111101':'11111111',
        '01111110':'11110011', '01111111':'11010010', '10000000':'11001101', '10000001':'00001100', '10000010':'00010011', '10000011':'11101100',
        '10000100':'01011111', '10000101':'10010111', '10000110':'01000100', '10000111':'00010111', '10001000':'11000100', '10001001':'10100111',
        '10001010':'01111110', '10001011':'00111101', '10001100':'01100100', '10001101':'01011101', '10001110':'00011001', '10001111':'01110011',
        '10010000':'01100000', '10010001':'10000001', '10010010':'01001111', '10010011':'11011100', '10010100':'00100010', '10010101':'00101010',
        '10010110':'10010000', '10010111':'10001000', '10011000':'01000110', '10011001':'11101110', '10011010':'10111000', '10011011':'00010100',
        '10011100':'11011110', '10011101':'01011110', '10011110':'00001011', '10011111':'11011011', '10100000':'11100000', '10100001':'00110010',
        '10100010':'00111010', '10100011':'00001010', '10100100':'01001001', '10100101':'00000110', '10100110':'00100100', '10100111':'01011100',
        '10101000':'11000010', '10101001':'11010011', '10101010':'10101100', '10101011':'01100010', '10101100':'10010001', '10101101':'10010101',
        '10101110':'11100100', '10101111':'01111001', '10110000':'11100111', '10110001':'11001000', '10110010':'00110111', '10110011':'01101101',
        '10110100':'10001101', '10110101':'11010101', '10110110':'01001110', '10110111':'10101001', '10111000':'01101100', '10111001':'01010110',
        '10111010':'11110100', '10111011':'11101010', '10111100':'01100101', '10111101':'01111010', '10111110':'10101110', '10111111':'00001000',
        '11000000':'10111010', '11000001':'01111000', '11000010':'00100101', '11000011':'00101110', '11000100':'00011100', '11000101':'10100110',
        '11000110':'10110100', '11000111':'11000110', '11001000':'11101000', '11001001':'11011101', '11001010':'01110100', '11001011':'00011111',
        '11001100':'01001011', '11001101':'10111101', '11001110':'10001011', '11001111':'10001010', '11010000':'01110000', '11010001':'00111110',
        '11010010':'10110101', '11010011':'01100110', '11010100':'01001000', '11010101':'00000011', '11010110':'11110110', '11010111':'00001110',
        '11011000':'01100001', '11011001':'00110101', '11011010':'01010111', '11011011':'10111001', '11011100':'10000110', '11011101':'11000001',
        '11011110':'00011101', '11011111':'10011110', '11100000':'11100001', '11100001':'11111000', '11100010':'10011000', '11100011':'00010001',
        '11100100':'01101001', '11100101':'11011001', '11100110':'10001110', '11100111':'10010100', '11101000':'10011011', '11101001':'00011110',
        '11101010':'10000111', '11101011':'11101001', '11101100':'11001110', '11101101':'01010101', '11101110':'00101000', '11101111':'11011111',
        '11110000':'10001100', '11110001':'10100001', '11110010':'10001001', '11110011':'00001101', '11110100':'10111111', '11110101':'11100110',
        '11110110':'01000010', '11110111':'01101000', '11111000':'01000001', '11111001':'10011001', '11111010':'00101101', '11111011':'00001111',
        '11111100':'10110000', '11111101':'01010100', '11111110':'10111011', '11111111':'00010110'}

print('16文字以下のものを入力してください。')                                                # 注意書き
print('もし、16文字に満たない場合は、便宜上0を必要な数だけ補い、16文字に揃えてください。')
print('')

mojiretsu = input('入力（平文） : ')   # 平文を取得
print('')
nagasa = len(mojiretsu)

if (nagasa > 16):     # 128bitを超える場合はエラー文を出力して強制終了
    print('エラー : 入力できる文字数の上限を超えました。')
    sys.exit()

list_1 = []
for i in range(nagasa):                   # 1文字ずつlist_1に入れていく
    list_1.append(mojiretsu[i])

for i in range(nagasa):                   # ASCIIコード(10進数)に変換
    list_1[i] = ord(list_1[i])

for i in range(nagasa):                   # 2進数に変換(0bを付けない形式)
    list_1[i] = format(list_1[i], 'b')

# 2進数に変換した後、
# 8桁に満たない場合は先頭に0を付けて8桁に揃える
for i in range(len(list_1)):
    if (len(list_1[i]) != 8):
        for j in range(8-len(list_1[i])):
            list_1[i] = '0' + list_1[i]

binmoji = ''.join(list_1)          # join関数でlist_1の各要素を文字列として結合

nagasa_2 = len(binmoji)            # 2進数化した文字列の長さ

hirabun = binmoji     # 後で使う用に、2進数化した平文を変数に保存

# 暗号化・復号化の確認用の平文
hirabun = '00000000000100010010001000110011010001000101010101100110011101111000100010011001101010101011101111001100110111011110111011111111'

print('平文の、ASCIIコードによる2進数化')
print(binmoji)
print('')

ransu_list = []
for i in range(128):
    ransu_list.append(str(random.randint(0, 1)))
k0 = ''.join(ransu_list)

# 以下のk0はサブ鍵の確認用
#k0 = '00101011011111100001010100010110001010001010111011010010101001101010101111110111000101011000100000001001110011110100111100111100'
# 以下のk0は暗号化・復号化の確認用
k0 = '00000000000000010000001000000011000001000000010100000110000001110000100000001001000010100000101100001100000011010000111000001111'


binmoji = k0

###################################
# サブ鍵生成アルゴリズム
###################################

print('サブ鍵生成を実行します。')

# Rconを先に定義するする
rcon_1 = '00000001000000000000000000000000'
rcon_2 = '00000010000000000000000000000000'
rcon_3 = '00000100000000000000000000000000'
rcon_4 = '00001000000000000000000000000000'
rcon_5 = '00010000000000000000000000000000'
rcon_6 = '00100000000000000000000000000000'
rcon_7 = '01000000000000000000000000000000'
rcon_8 = '10000000000000000000000000000000'
rcon_9 = '00011011000000000000000000000000'
rcon_10 = '00110110000000000000000000000000'

##########
# i = 1
##########
# シフトする以前のw3を取得
w3_mae = []
for i in range(96, 128):
    w3_mae.append(binmoji[i])
w3_mae = ''.join(w3_mae)

# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(104, 128):     
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 104):     
    binmoji_shift_list.append(binmoji[i])

# 現状 : binmoji_shift_listの要素数は128で、シフト実行後のbinmojiが一つずつ格納されているイメージ

binmoji_shift = ''.join(binmoji_shift_list)    # binmoji_shift_listを文字列として結合

# 後で消す
print('')
print('RotWordsの直後')
emp_list = []
for i in range(96, 128):
    emp_list.append(binmoji_shift_list[i])
rot = ''.join(emp_list)
print(rot)                                               #######################################################################################################

# w3の各ブロックををsboxにより置換
# w3_0について
okikae_0 = []
for i in range(96, 104):     # w3_0の要素(8bit)を取得
    okikae_0.append(binmoji_shift[i])
okikae_moji_0 = ''.join(okikae_0)     
after_okikae_moji_0 = sbox[okikae_moji_0]   # w3_0の要素(8bit)をsboxで置換した文字列

# w3_1について
okikae_1 = []
for i in range(104, 112):
    okikae_1.append(binmoji_shift[i])
okikae_moji_1 = ''.join(okikae_1)
after_okikae_moji_1 = sbox[okikae_moji_1]

# w3_2について
okikae_2 = []
for i in range(112, 120):
    okikae_2.append(binmoji_shift[i])
okikae_moji_2 = ''.join(okikae_2)
after_okikae_moji_2 = sbox[okikae_moji_2]

# w3_3について
okikae_3 = []
for i in range(120, 128):
    okikae_3.append(binmoji_shift[i])
okikae_moji_3 = ''.join(okikae_3)
after_okikae_moji_3 = sbox[okikae_moji_3]

# 置き換え後ののw3を文字列として結合(32bit)
after_okikae_moji = after_okikae_moji_0 + after_okikae_moji_1 + after_okikae_moji_2 + after_okikae_moji_3
#####################################################################################################################################後で消す
print('SubWordsの直後')
print(after_okikae_moji)

# 置き換え後の128bit全体
after_okikae_128 = []
for i in range(0, 96):     # w0～w2は変更なしなので、そのまま取得
    after_okikae_128.append(binmoji_shift_list[i])
for i in range(0, 32):     # w3は、after_okikae_mojiを取得
    after_okikae_128.append(after_okikae_moji[i])

after_okikae_128_moji = ''.join(after_okikae_128)

# w0の取得
emp_list = []
for i in range(0, 32):
    emp_list.append(after_okikae_128_moji[i])
w0 = ''.join(emp_list)
# w1の取得
emp_list = []
for i in range(32, 64):
    emp_list.append(after_okikae_128_moji[i])
w1 = ''.join(emp_list)
# w2の取得
emp_list = []
for i in range(64, 96):
    emp_list.append(after_okikae_128_moji[i])
w2 = ''.join(emp_list)
# w3の取得
emp_list = []
for i in range(96, 128):
    emp_list.append(after_okikae_128_moji[i])
w3 = ''.join(emp_list)

# 排他的論理和(i=1)
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_1, after_okikae_moji)]
xor_1_new = []
for i in range(32):
    xor_1_new.append(str(xor_1[i]))
temp = ''.join(xor_1_new)
########################################################################################################################後で消す
print('tempの値')
print(temp)

# ステップ4c(テキスト)
# j=0
w4_list = [ord(a) ^ ord(b) for a,b in zip(w0, temp)]
w4_list_str = []
for i in range(32):
    w4_list_str.append(str(w4_list[i]))
w4 = ''.join(w4_list_str)
temp = w4

# j=1
w5_list = [ord(a) ^ ord(b) for a,b in zip(w1, temp)]
w5_list_str = []
for i in range(32):
    w5_list_str.append(str(w5_list[i]))
w5 = ''.join(w5_list_str)
temp = w5

# j=2
w6_list = [ord(a) ^ ord(b) for a,b in zip(w2, temp)]
w6_list_str = []
for i in range(32):
    w6_list_str.append(str(w6_list[i]))
w6 = ''.join(w6_list_str)
temp = w6

# j=3
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3_mae, temp)]
w7_list_str = []
for i in range(32):
    w7_list_str.append(str(w7_list[i]))
w7 = ''.join(w7_list_str)
temp = w7

k1 = w4 + w5 + w6 + w7

print('1つ目のサブ鍵 k1')
print(k1)

binmoji = k1     # binmojiを更新


##########
# i = 2
##########
# シフトする以前のw3を取得
w3_mae = []
for i in range(96, 128):
    w3_mae.append(binmoji[i])
w3_mae = ''.join(w3_mae)

# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(104, 128):     
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 104):     
    binmoji_shift_list.append(binmoji[i])

# 後で消す ####################################################################################################################################

# 現状 : binmoji_shift_listの要素数は128で、シフト実行後のbinmojiが一つずつ格納されているイメージ

binmoji_shift = ''.join(binmoji_shift_list)    # binmoji_shift_listを文字列として結合

emp_list = []
for i in range(96, 128):
    emp_list.append(binmoji_shift_list[i])
rot2 = ''.join(emp_list)
print('RotWordsの直後(i=2)')
print(rot2)

# w3の各ブロックををsboxにより置換
# w3_0について
okikae_0 = []
for i in range(96, 104):     # w3_0の要素(8bit)を取得
    okikae_0.append(binmoji_shift[i])
okikae_moji_0 = ''.join(okikae_0)     
after_okikae_moji_0 = sbox[okikae_moji_0]   # w3_0の要素(8bit)をsboxで置換した文字列

# w3_1について
okikae_1 = []
for i in range(104, 112):
    okikae_1.append(binmoji_shift[i])
okikae_moji_1 = ''.join(okikae_1)
after_okikae_moji_1 = sbox[okikae_moji_1]

# w3_2について
okikae_2 = []
for i in range(112, 120):
    okikae_2.append(binmoji_shift[i])
okikae_moji_2 = ''.join(okikae_2)
after_okikae_moji_2 = sbox[okikae_moji_2]

# w3_3について
okikae_3 = []
for i in range(120, 128):
    okikae_3.append(binmoji_shift[i])
okikae_moji_3 = ''.join(okikae_3)
after_okikae_moji_3 = sbox[okikae_moji_3]

# 置き換え後ののw3を文字列として結合(32bit)
after_okikae_moji = after_okikae_moji_0 + after_okikae_moji_1 + after_okikae_moji_2 + after_okikae_moji_3

# 置き換え後の128bit全体
after_okikae_128 = []
for i in range(0, 96):     # w0～w2は変更なしなので、そのまま取得
    after_okikae_128.append(binmoji_shift_list[i])
for i in range(0, 32):     # w3は、after_okikae_mojiを取得
    after_okikae_128.append(after_okikae_moji[i])

after_okikae_128_moji = ''.join(after_okikae_128)

# w0の取得
emp_list = []
for i in range(0, 32):
    emp_list.append(after_okikae_128_moji[i])
w0 = ''.join(emp_list)
# w1の取得
emp_list = []
for i in range(32, 64):
    emp_list.append(after_okikae_128_moji[i])
w1 = ''.join(emp_list)
# w2の取得
emp_list = []
for i in range(64, 96):
    emp_list.append(after_okikae_128_moji[i])
w2 = ''.join(emp_list)
# w3の取得
emp_list = []
for i in range(96, 128):
    emp_list.append(after_okikae_128_moji[i])
w3 = ''.join(emp_list)

# 排他的論理和
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_2, after_okikae_moji)]
xor_1_new = []
for i in range(32):
    xor_1_new.append(str(xor_1[i]))
temp = ''.join(xor_1_new)

# ステップ4c(テキスト)
# j=0
w4_list = [ord(a) ^ ord(b) for a,b in zip(w0, temp)]
w4_list_str = []
for i in range(32):
    w4_list_str.append(str(w4_list[i]))
w4 = ''.join(w4_list_str)
temp = w4

# j=1
w5_list = [ord(a) ^ ord(b) for a,b in zip(w1, temp)]
w5_list_str = []
for i in range(32):
    w5_list_str.append(str(w5_list[i]))
w5 = ''.join(w5_list_str)
temp = w5

# j=2
w6_list = [ord(a) ^ ord(b) for a,b in zip(w2, temp)]
w6_list_str = []
for i in range(32):
    w6_list_str.append(str(w6_list[i]))
w6 = ''.join(w6_list_str)
temp = w6

# j=3
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3_mae, temp)]
w7_list_str = []
for i in range(32):
    w7_list_str.append(str(w7_list[i]))
w7 = ''.join(w7_list_str)
temp = w7

k2 = w4 + w5 + w6 + w7
binmoji = k2     # binmojiの更新

print('2つ目のサブ鍵 k2')
print(k2)


##########
# i = 3
##########
# シフトする以前のw3を取得
w3_mae = []
for i in range(96, 128):
    w3_mae.append(binmoji[i])
w3_mae = ''.join(w3_mae)

# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(104, 128):     
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 104):     
    binmoji_shift_list.append(binmoji[i])

# 後で消す ####################################################################################################################################

# 現状 : binmoji_shift_listの要素数は128で、シフト実行後のbinmojiが一つずつ格納されているイメージ

binmoji_shift = ''.join(binmoji_shift_list)    # binmoji_shift_listを文字列として結合

# w3の各ブロックををsboxにより置換
# w3_0について
okikae_0 = []
for i in range(96, 104):     # w3_0の要素(8bit)を取得
    okikae_0.append(binmoji_shift[i])
okikae_moji_0 = ''.join(okikae_0)     
after_okikae_moji_0 = sbox[okikae_moji_0]   # w3_0の要素(8bit)をsboxで置換した文字列

# w3_1について
okikae_1 = []
for i in range(104, 112):
    okikae_1.append(binmoji_shift[i])
okikae_moji_1 = ''.join(okikae_1)
after_okikae_moji_1 = sbox[okikae_moji_1]

# w3_2について
okikae_2 = []
for i in range(112, 120):
    okikae_2.append(binmoji_shift[i])
okikae_moji_2 = ''.join(okikae_2)
after_okikae_moji_2 = sbox[okikae_moji_2]

# w3_3について
okikae_3 = []
for i in range(120, 128):
    okikae_3.append(binmoji_shift[i])
okikae_moji_3 = ''.join(okikae_3)
after_okikae_moji_3 = sbox[okikae_moji_3]

# 置き換え後ののw3を文字列として結合(32bit)
after_okikae_moji = after_okikae_moji_0 + after_okikae_moji_1 + after_okikae_moji_2 + after_okikae_moji_3

# 置き換え後の128bit全体
after_okikae_128 = []
for i in range(0, 96):     # w0～w2は変更なしなので、そのまま取得
    after_okikae_128.append(binmoji_shift_list[i])
for i in range(0, 32):     # w3は、after_okikae_mojiを取得
    after_okikae_128.append(after_okikae_moji[i])

after_okikae_128_moji = ''.join(after_okikae_128)

# w0の取得
emp_list = []
for i in range(0, 32):
    emp_list.append(after_okikae_128_moji[i])
w0 = ''.join(emp_list)
# w1の取得
emp_list = []
for i in range(32, 64):
    emp_list.append(after_okikae_128_moji[i])
w1 = ''.join(emp_list)
# w2の取得
emp_list = []
for i in range(64, 96):
    emp_list.append(after_okikae_128_moji[i])
w2 = ''.join(emp_list)
# w3の取得
emp_list = []
for i in range(96, 128):
    emp_list.append(after_okikae_128_moji[i])
w3 = ''.join(emp_list)

# 排他的論理和
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_3, after_okikae_moji)]
xor_1_new = []
for i in range(32):
    xor_1_new.append(str(xor_1[i]))
temp = ''.join(xor_1_new)

# ステップ4c(テキスト)
# j=0
w4_list = [ord(a) ^ ord(b) for a,b in zip(w0, temp)]
w4_list_str = []
for i in range(32):
    w4_list_str.append(str(w4_list[i]))
w4 = ''.join(w4_list_str)
temp = w4

# j=1
w5_list = [ord(a) ^ ord(b) for a,b in zip(w1, temp)]
w5_list_str = []
for i in range(32):
    w5_list_str.append(str(w5_list[i]))
w5 = ''.join(w5_list_str)
temp = w5

# j=2
w6_list = [ord(a) ^ ord(b) for a,b in zip(w2, temp)]
w6_list_str = []
for i in range(32):
    w6_list_str.append(str(w6_list[i]))
w6 = ''.join(w6_list_str)
temp = w6

# j=3
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3_mae, temp)]
w7_list_str = []
for i in range(32):
    w7_list_str.append(str(w7_list[i]))
w7 = ''.join(w7_list_str)
temp = w7

k3 = w4 + w5 + w6 + w7
binmoji = k3     # binmojiの更新

print('3つ目のサブ鍵 k3')
print(k3)


##########
# i = 4
##########
# シフトする以前のw3を取得
w3_mae = []
for i in range(96, 128):
    w3_mae.append(binmoji[i])
w3_mae = ''.join(w3_mae)

# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(104, 128):     
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 104):     
    binmoji_shift_list.append(binmoji[i])

# 後で消す ####################################################################################################################################

# 現状 : binmoji_shift_listの要素数は128で、シフト実行後のbinmojiが一つずつ格納されているイメージ

binmoji_shift = ''.join(binmoji_shift_list)    # binmoji_shift_listを文字列として結合


# w3の各ブロックををsboxにより置換
# w3_0について
okikae_0 = []
for i in range(96, 104):     # w3_0の要素(8bit)を取得
    okikae_0.append(binmoji_shift[i])
okikae_moji_0 = ''.join(okikae_0)     
after_okikae_moji_0 = sbox[okikae_moji_0]   # w3_0の要素(8bit)をsboxで置換した文字列

# w3_1について
okikae_1 = []
for i in range(104, 112):
    okikae_1.append(binmoji_shift[i])
okikae_moji_1 = ''.join(okikae_1)
after_okikae_moji_1 = sbox[okikae_moji_1]

# w3_2について
okikae_2 = []
for i in range(112, 120):
    okikae_2.append(binmoji_shift[i])
okikae_moji_2 = ''.join(okikae_2)
after_okikae_moji_2 = sbox[okikae_moji_2]

# w3_3について
okikae_3 = []
for i in range(120, 128):
    okikae_3.append(binmoji_shift[i])
okikae_moji_3 = ''.join(okikae_3)
after_okikae_moji_3 = sbox[okikae_moji_3]

# 置き換え後ののw3を文字列として結合(32bit)
after_okikae_moji = after_okikae_moji_0 + after_okikae_moji_1 + after_okikae_moji_2 + after_okikae_moji_3

# 置き換え後の128bit全体
after_okikae_128 = []
for i in range(0, 96):     # w0～w2は変更なしなので、そのまま取得
    after_okikae_128.append(binmoji_shift_list[i])
for i in range(0, 32):     # w3は、after_okikae_mojiを取得
    after_okikae_128.append(after_okikae_moji[i])

after_okikae_128_moji = ''.join(after_okikae_128)

# w0の取得
emp_list = []
for i in range(0, 32):
    emp_list.append(after_okikae_128_moji[i])
w0 = ''.join(emp_list)
# w1の取得
emp_list = []
for i in range(32, 64):
    emp_list.append(after_okikae_128_moji[i])
w1 = ''.join(emp_list)
# w2の取得
emp_list = []
for i in range(64, 96):
    emp_list.append(after_okikae_128_moji[i])
w2 = ''.join(emp_list)
# w3の取得
emp_list = []
for i in range(96, 128):
    emp_list.append(after_okikae_128_moji[i])
w3 = ''.join(emp_list)

# 排他的論理和
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_4, after_okikae_moji)]
xor_1_new = []
for i in range(32):
    xor_1_new.append(str(xor_1[i]))
temp = ''.join(xor_1_new)

# ステップ4c(テキスト)
# j=0
w4_list = [ord(a) ^ ord(b) for a,b in zip(w0, temp)]
w4_list_str = []
for i in range(32):
    w4_list_str.append(str(w4_list[i]))
w4 = ''.join(w4_list_str)
temp = w4

# j=1
w5_list = [ord(a) ^ ord(b) for a,b in zip(w1, temp)]
w5_list_str = []
for i in range(32):
    w5_list_str.append(str(w5_list[i]))
w5 = ''.join(w5_list_str)
temp = w5

# j=2
w6_list = [ord(a) ^ ord(b) for a,b in zip(w2, temp)]
w6_list_str = []
for i in range(32):
    w6_list_str.append(str(w6_list[i]))
w6 = ''.join(w6_list_str)
temp = w6

# j=3
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3_mae, temp)]
w7_list_str = []
for i in range(32):
    w7_list_str.append(str(w7_list[i]))
w7 = ''.join(w7_list_str)
temp = w7

k4 = w4 + w5 + w6 + w7
binmoji = k4     # binmojiの更新

print('4つ目のサブ鍵 k4')
print(k4)


##########
# i = 5
##########
# シフトする以前のw3を取得
w3_mae = []
for i in range(96, 128):
    w3_mae.append(binmoji[i])
w3_mae = ''.join(w3_mae)

# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(104, 128):     
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 104):     
    binmoji_shift_list.append(binmoji[i])

# 後で消す ####################################################################################################################################

# 現状 : binmoji_shift_listの要素数は128で、シフト実行後のbinmojiが一つずつ格納されているイメージ

binmoji_shift = ''.join(binmoji_shift_list)    # binmoji_shift_listを文字列として結合


# w3の各ブロックををsboxにより置換
# w3_0について
okikae_0 = []
for i in range(96, 104):     # w3_0の要素(8bit)を取得
    okikae_0.append(binmoji_shift[i])
okikae_moji_0 = ''.join(okikae_0)     
after_okikae_moji_0 = sbox[okikae_moji_0]   # w3_0の要素(8bit)をsboxで置換した文字列

# w3_1について
okikae_1 = []
for i in range(104, 112):
    okikae_1.append(binmoji_shift[i])
okikae_moji_1 = ''.join(okikae_1)
after_okikae_moji_1 = sbox[okikae_moji_1]

# w3_2について
okikae_2 = []
for i in range(112, 120):
    okikae_2.append(binmoji_shift[i])
okikae_moji_2 = ''.join(okikae_2)
after_okikae_moji_2 = sbox[okikae_moji_2]

# w3_3について
okikae_3 = []
for i in range(120, 128):
    okikae_3.append(binmoji_shift[i])
okikae_moji_3 = ''.join(okikae_3)
after_okikae_moji_3 = sbox[okikae_moji_3]

# 置き換え後ののw3を文字列として結合(32bit)
after_okikae_moji = after_okikae_moji_0 + after_okikae_moji_1 + after_okikae_moji_2 + after_okikae_moji_3

# 置き換え後の128bit全体
after_okikae_128 = []
for i in range(0, 96):     # w0～w2は変更なしなので、そのまま取得
    after_okikae_128.append(binmoji_shift_list[i])
for i in range(0, 32):     # w3は、after_okikae_mojiを取得
    after_okikae_128.append(after_okikae_moji[i])

after_okikae_128_moji = ''.join(after_okikae_128)

# w0の取得
emp_list = []
for i in range(0, 32):
    emp_list.append(after_okikae_128_moji[i])
w0 = ''.join(emp_list)
# w1の取得
emp_list = []
for i in range(32, 64):
    emp_list.append(after_okikae_128_moji[i])
w1 = ''.join(emp_list)
# w2の取得
emp_list = []
for i in range(64, 96):
    emp_list.append(after_okikae_128_moji[i])
w2 = ''.join(emp_list)
# w3の取得
emp_list = []
for i in range(96, 128):
    emp_list.append(after_okikae_128_moji[i])
w3 = ''.join(emp_list)

# 排他的論理和
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_5, after_okikae_moji)]
xor_1_new = []
for i in range(32):
    xor_1_new.append(str(xor_1[i]))
temp = ''.join(xor_1_new)

# ステップ4c(テキスト)
# j=0
w4_list = [ord(a) ^ ord(b) for a,b in zip(w0, temp)]
w4_list_str = []
for i in range(32):
    w4_list_str.append(str(w4_list[i]))
w4 = ''.join(w4_list_str)
temp = w4

# j=1
w5_list = [ord(a) ^ ord(b) for a,b in zip(w1, temp)]
w5_list_str = []
for i in range(32):
    w5_list_str.append(str(w5_list[i]))
w5 = ''.join(w5_list_str)
temp = w5

# j=2
w6_list = [ord(a) ^ ord(b) for a,b in zip(w2, temp)]
w6_list_str = []
for i in range(32):
    w6_list_str.append(str(w6_list[i]))
w6 = ''.join(w6_list_str)
temp = w6

# j=3
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3_mae, temp)]
w7_list_str = []
for i in range(32):
    w7_list_str.append(str(w7_list[i]))
w7 = ''.join(w7_list_str)
temp = w7

k5 = w4 + w5 + w6 + w7
binmoji = k5     # binmojiの更新

print('5つ目のサブ鍵 k5')
print(k5)


##########
# i = 6
##########
# シフトする以前のw3を取得
w3_mae = []
for i in range(96, 128):
    w3_mae.append(binmoji[i])
w3_mae = ''.join(w3_mae)

# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(104, 128):     
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 104):     
    binmoji_shift_list.append(binmoji[i])

# 後で消す ####################################################################################################################################

# 現状 : binmoji_shift_listの要素数は128で、シフト実行後のbinmojiが一つずつ格納されているイメージ

binmoji_shift = ''.join(binmoji_shift_list)    # binmoji_shift_listを文字列として結合


# w3の各ブロックををsboxにより置換
# w3_0について
okikae_0 = []
for i in range(96, 104):     # w3_0の要素(8bit)を取得
    okikae_0.append(binmoji_shift[i])
okikae_moji_0 = ''.join(okikae_0)     
after_okikae_moji_0 = sbox[okikae_moji_0]   # w3_0の要素(8bit)をsboxで置換した文字列

# w3_1について
okikae_1 = []
for i in range(104, 112):
    okikae_1.append(binmoji_shift[i])
okikae_moji_1 = ''.join(okikae_1)
after_okikae_moji_1 = sbox[okikae_moji_1]

# w3_2について
okikae_2 = []
for i in range(112, 120):
    okikae_2.append(binmoji_shift[i])
okikae_moji_2 = ''.join(okikae_2)
after_okikae_moji_2 = sbox[okikae_moji_2]

# w3_3について
okikae_3 = []
for i in range(120, 128):
    okikae_3.append(binmoji_shift[i])
okikae_moji_3 = ''.join(okikae_3)
after_okikae_moji_3 = sbox[okikae_moji_3]

# 置き換え後ののw3を文字列として結合(32bit)
after_okikae_moji = after_okikae_moji_0 + after_okikae_moji_1 + after_okikae_moji_2 + after_okikae_moji_3

# 置き換え後の128bit全体
after_okikae_128 = []
for i in range(0, 96):     # w0～w2は変更なしなので、そのまま取得
    after_okikae_128.append(binmoji_shift_list[i])
for i in range(0, 32):     # w3は、after_okikae_mojiを取得
    after_okikae_128.append(after_okikae_moji[i])

after_okikae_128_moji = ''.join(after_okikae_128)

# w0の取得
emp_list = []
for i in range(0, 32):
    emp_list.append(after_okikae_128_moji[i])
w0 = ''.join(emp_list)
# w1の取得
emp_list = []
for i in range(32, 64):
    emp_list.append(after_okikae_128_moji[i])
w1 = ''.join(emp_list)
# w2の取得
emp_list = []
for i in range(64, 96):
    emp_list.append(after_okikae_128_moji[i])
w2 = ''.join(emp_list)
# w3の取得
emp_list = []
for i in range(96, 128):
    emp_list.append(after_okikae_128_moji[i])
w3 = ''.join(emp_list)

# 排他的論理和
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_6, after_okikae_moji)]
xor_1_new = []
for i in range(32):
    xor_1_new.append(str(xor_1[i]))
temp = ''.join(xor_1_new)

# ステップ4c(テキスト)
# j=0
w4_list = [ord(a) ^ ord(b) for a,b in zip(w0, temp)]
w4_list_str = []
for i in range(32):
    w4_list_str.append(str(w4_list[i]))
w4 = ''.join(w4_list_str)
temp = w4

# j=1
w5_list = [ord(a) ^ ord(b) for a,b in zip(w1, temp)]
w5_list_str = []
for i in range(32):
    w5_list_str.append(str(w5_list[i]))
w5 = ''.join(w5_list_str)
temp = w5

# j=2
w6_list = [ord(a) ^ ord(b) for a,b in zip(w2, temp)]
w6_list_str = []
for i in range(32):
    w6_list_str.append(str(w6_list[i]))
w6 = ''.join(w6_list_str)
temp = w6

# j=3
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3_mae, temp)]
w7_list_str = []
for i in range(32):
    w7_list_str.append(str(w7_list[i]))
w7 = ''.join(w7_list_str)
temp = w7

k6 = w4 + w5 + w6 + w7
binmoji = k6     # binmojiの更新

print('6つ目のサブ鍵 k6')
print(k6)


##########
# i = 7
##########
# シフトする以前のw3を取得
w3_mae = []
for i in range(96, 128):
    w3_mae.append(binmoji[i])
w3_mae = ''.join(w3_mae)

# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(104, 128):     
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 104):     
    binmoji_shift_list.append(binmoji[i])

# 後で消す ####################################################################################################################################

# 現状 : binmoji_shift_listの要素数は128で、シフト実行後のbinmojiが一つずつ格納されているイメージ

binmoji_shift = ''.join(binmoji_shift_list)    # binmoji_shift_listを文字列として結合


# w3の各ブロックををsboxにより置換
# w3_0について
okikae_0 = []
for i in range(96, 104):     # w3_0の要素(8bit)を取得
    okikae_0.append(binmoji_shift[i])
okikae_moji_0 = ''.join(okikae_0)     
after_okikae_moji_0 = sbox[okikae_moji_0]   # w3_0の要素(8bit)をsboxで置換した文字列

# w3_1について
okikae_1 = []
for i in range(104, 112):
    okikae_1.append(binmoji_shift[i])
okikae_moji_1 = ''.join(okikae_1)
after_okikae_moji_1 = sbox[okikae_moji_1]

# w3_2について
okikae_2 = []
for i in range(112, 120):
    okikae_2.append(binmoji_shift[i])
okikae_moji_2 = ''.join(okikae_2)
after_okikae_moji_2 = sbox[okikae_moji_2]

# w3_3について
okikae_3 = []
for i in range(120, 128):
    okikae_3.append(binmoji_shift[i])
okikae_moji_3 = ''.join(okikae_3)
after_okikae_moji_3 = sbox[okikae_moji_3]

# 置き換え後ののw3を文字列として結合(32bit)
after_okikae_moji = after_okikae_moji_0 + after_okikae_moji_1 + after_okikae_moji_2 + after_okikae_moji_3

# 置き換え後の128bit全体
after_okikae_128 = []
for i in range(0, 96):     # w0～w2は変更なしなので、そのまま取得
    after_okikae_128.append(binmoji_shift_list[i])
for i in range(0, 32):     # w3は、after_okikae_mojiを取得
    after_okikae_128.append(after_okikae_moji[i])

after_okikae_128_moji = ''.join(after_okikae_128)

# w0の取得
emp_list = []
for i in range(0, 32):
    emp_list.append(after_okikae_128_moji[i])
w0 = ''.join(emp_list)
# w1の取得
emp_list = []
for i in range(32, 64):
    emp_list.append(after_okikae_128_moji[i])
w1 = ''.join(emp_list)
# w2の取得
emp_list = []
for i in range(64, 96):
    emp_list.append(after_okikae_128_moji[i])
w2 = ''.join(emp_list)
# w3の取得
emp_list = []
for i in range(96, 128):
    emp_list.append(after_okikae_128_moji[i])
w3 = ''.join(emp_list)

# 排他的論理和
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_7, after_okikae_moji)]
xor_1_new = []
for i in range(32):
    xor_1_new.append(str(xor_1[i]))
temp = ''.join(xor_1_new)

# ステップ4c(テキスト)
# j=0
w4_list = [ord(a) ^ ord(b) for a,b in zip(w0, temp)]
w4_list_str = []
for i in range(32):
    w4_list_str.append(str(w4_list[i]))
w4 = ''.join(w4_list_str)
temp = w4

# j=1
w5_list = [ord(a) ^ ord(b) for a,b in zip(w1, temp)]
w5_list_str = []
for i in range(32):
    w5_list_str.append(str(w5_list[i]))
w5 = ''.join(w5_list_str)
temp = w5

# j=2
w6_list = [ord(a) ^ ord(b) for a,b in zip(w2, temp)]
w6_list_str = []
for i in range(32):
    w6_list_str.append(str(w6_list[i]))
w6 = ''.join(w6_list_str)
temp = w6

# j=3
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3_mae, temp)]
w7_list_str = []
for i in range(32):
    w7_list_str.append(str(w7_list[i]))
w7 = ''.join(w7_list_str)
temp = w7

k7 = w4 + w5 + w6 + w7
binmoji = k7     # binmojiの更新

print('7つ目のサブ鍵 k7')
print(k7)


##########
# i = 8
##########
# シフトする以前のw3を取得
w3_mae = []
for i in range(96, 128):
    w3_mae.append(binmoji[i])
w3_mae = ''.join(w3_mae)

# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(104, 128):     
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 104):     
    binmoji_shift_list.append(binmoji[i])

# 後で消す ####################################################################################################################################

# 現状 : binmoji_shift_listの要素数は128で、シフト実行後のbinmojiが一つずつ格納されているイメージ

binmoji_shift = ''.join(binmoji_shift_list)    # binmoji_shift_listを文字列として結合


# w3の各ブロックををsboxにより置換
# w3_0について
okikae_0 = []
for i in range(96, 104):     # w3_0の要素(8bit)を取得
    okikae_0.append(binmoji_shift[i])
okikae_moji_0 = ''.join(okikae_0)     
after_okikae_moji_0 = sbox[okikae_moji_0]   # w3_0の要素(8bit)をsboxで置換した文字列

# w3_1について
okikae_1 = []
for i in range(104, 112):
    okikae_1.append(binmoji_shift[i])
okikae_moji_1 = ''.join(okikae_1)
after_okikae_moji_1 = sbox[okikae_moji_1]

# w3_2について
okikae_2 = []
for i in range(112, 120):
    okikae_2.append(binmoji_shift[i])
okikae_moji_2 = ''.join(okikae_2)
after_okikae_moji_2 = sbox[okikae_moji_2]

# w3_3について
okikae_3 = []
for i in range(120, 128):
    okikae_3.append(binmoji_shift[i])
okikae_moji_3 = ''.join(okikae_3)
after_okikae_moji_3 = sbox[okikae_moji_3]

# 置き換え後ののw3を文字列として結合(32bit)
after_okikae_moji = after_okikae_moji_0 + after_okikae_moji_1 + after_okikae_moji_2 + after_okikae_moji_3

# 置き換え後の128bit全体
after_okikae_128 = []
for i in range(0, 96):     # w0～w2は変更なしなので、そのまま取得
    after_okikae_128.append(binmoji_shift_list[i])
for i in range(0, 32):     # w3は、after_okikae_mojiを取得
    after_okikae_128.append(after_okikae_moji[i])

after_okikae_128_moji = ''.join(after_okikae_128)

# w0の取得
emp_list = []
for i in range(0, 32):
    emp_list.append(after_okikae_128_moji[i])
w0 = ''.join(emp_list)
# w1の取得
emp_list = []
for i in range(32, 64):
    emp_list.append(after_okikae_128_moji[i])
w1 = ''.join(emp_list)
# w2の取得
emp_list = []
for i in range(64, 96):
    emp_list.append(after_okikae_128_moji[i])
w2 = ''.join(emp_list)
# w3の取得
emp_list = []
for i in range(96, 128):
    emp_list.append(after_okikae_128_moji[i])
w3 = ''.join(emp_list)

# 排他的論理和
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_8, after_okikae_moji)]
xor_1_new = []
for i in range(32):
    xor_1_new.append(str(xor_1[i]))
temp = ''.join(xor_1_new)

# ステップ4c(テキスト)
# j=0
w4_list = [ord(a) ^ ord(b) for a,b in zip(w0, temp)]
w4_list_str = []
for i in range(32):
    w4_list_str.append(str(w4_list[i]))
w4 = ''.join(w4_list_str)
temp = w4

# j=1
w5_list = [ord(a) ^ ord(b) for a,b in zip(w1, temp)]
w5_list_str = []
for i in range(32):
    w5_list_str.append(str(w5_list[i]))
w5 = ''.join(w5_list_str)
temp = w5

# j=2
w6_list = [ord(a) ^ ord(b) for a,b in zip(w2, temp)]
w6_list_str = []
for i in range(32):
    w6_list_str.append(str(w6_list[i]))
w6 = ''.join(w6_list_str)
temp = w6

# j=3
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3_mae, temp)]
w7_list_str = []
for i in range(32):
    w7_list_str.append(str(w7_list[i]))
w7 = ''.join(w7_list_str)
temp = w7

k8 = w4 + w5 + w6 + w7
binmoji = k8     # binmojiの更新

print('8つ目のサブ鍵 k8')
print(k8)


##########
# i = 9
##########
# シフトする以前のw3を取得
w3_mae = []
for i in range(96, 128):
    w3_mae.append(binmoji[i])
w3_mae = ''.join(w3_mae)

# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(104, 128):     
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 104):     
    binmoji_shift_list.append(binmoji[i])

# 後で消す ####################################################################################################################################

# 現状 : binmoji_shift_listの要素数は128で、シフト実行後のbinmojiが一つずつ格納されているイメージ

binmoji_shift = ''.join(binmoji_shift_list)    # binmoji_shift_listを文字列として結合


# w3の各ブロックををsboxにより置換
# w3_0について
okikae_0 = []
for i in range(96, 104):     # w3_0の要素(8bit)を取得
    okikae_0.append(binmoji_shift[i])
okikae_moji_0 = ''.join(okikae_0)     
after_okikae_moji_0 = sbox[okikae_moji_0]   # w3_0の要素(8bit)をsboxで置換した文字列

# w3_1について
okikae_1 = []
for i in range(104, 112):
    okikae_1.append(binmoji_shift[i])
okikae_moji_1 = ''.join(okikae_1)
after_okikae_moji_1 = sbox[okikae_moji_1]

# w3_2について
okikae_2 = []
for i in range(112, 120):
    okikae_2.append(binmoji_shift[i])
okikae_moji_2 = ''.join(okikae_2)
after_okikae_moji_2 = sbox[okikae_moji_2]

# w3_3について
okikae_3 = []
for i in range(120, 128):
    okikae_3.append(binmoji_shift[i])
okikae_moji_3 = ''.join(okikae_3)
after_okikae_moji_3 = sbox[okikae_moji_3]

# 置き換え後ののw3を文字列として結合(32bit)
after_okikae_moji = after_okikae_moji_0 + after_okikae_moji_1 + after_okikae_moji_2 + after_okikae_moji_3

# 置き換え後の128bit全体
after_okikae_128 = []
for i in range(0, 96):     # w0～w2は変更なしなので、そのまま取得
    after_okikae_128.append(binmoji_shift_list[i])
for i in range(0, 32):     # w3は、after_okikae_mojiを取得
    after_okikae_128.append(after_okikae_moji[i])

after_okikae_128_moji = ''.join(after_okikae_128)

# w0の取得
emp_list = []
for i in range(0, 32):
    emp_list.append(after_okikae_128_moji[i])
w0 = ''.join(emp_list)
# w1の取得
emp_list = []
for i in range(32, 64):
    emp_list.append(after_okikae_128_moji[i])
w1 = ''.join(emp_list)
# w2の取得
emp_list = []
for i in range(64, 96):
    emp_list.append(after_okikae_128_moji[i])
w2 = ''.join(emp_list)
# w3の取得
emp_list = []
for i in range(96, 128):
    emp_list.append(after_okikae_128_moji[i])
w3 = ''.join(emp_list)

# 排他的論理和
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_9, after_okikae_moji)]
xor_1_new = []
for i in range(32):
    xor_1_new.append(str(xor_1[i]))
temp = ''.join(xor_1_new)

# ステップ4c(テキスト)
# j=0
w4_list = [ord(a) ^ ord(b) for a,b in zip(w0, temp)]
w4_list_str = []
for i in range(32):
    w4_list_str.append(str(w4_list[i]))
w4 = ''.join(w4_list_str)
temp = w4

# j=1
w5_list = [ord(a) ^ ord(b) for a,b in zip(w1, temp)]
w5_list_str = []
for i in range(32):
    w5_list_str.append(str(w5_list[i]))
w5 = ''.join(w5_list_str)
temp = w5

# j=2
w6_list = [ord(a) ^ ord(b) for a,b in zip(w2, temp)]
w6_list_str = []
for i in range(32):
    w6_list_str.append(str(w6_list[i]))
w6 = ''.join(w6_list_str)
temp = w6

# j=3
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3_mae, temp)]
w7_list_str = []
for i in range(32):
    w7_list_str.append(str(w7_list[i]))
w7 = ''.join(w7_list_str)
temp = w7

k9 = w4 + w5 + w6 + w7
binmoji = k9     # binmojiの更新

print('9つ目のサブ鍵 k9')
print(k9)


##########
# i = 10
##########
# シフトする以前のw3を取得
w3_mae = []
for i in range(96, 128):
    w3_mae.append(binmoji[i])
w3_mae = ''.join(w3_mae)

# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(104, 128):     
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 104):     
    binmoji_shift_list.append(binmoji[i])

# 後で消す ####################################################################################################################################

# 現状 : binmoji_shift_listの要素数は128で、シフト実行後のbinmojiが一つずつ格納されているイメージ

binmoji_shift = ''.join(binmoji_shift_list)    # binmoji_shift_listを文字列として結合


# w3の各ブロックををsboxにより置換
# w3_0について
okikae_0 = []
for i in range(96, 104):     # w3_0の要素(8bit)を取得
    okikae_0.append(binmoji_shift[i])
okikae_moji_0 = ''.join(okikae_0)     
after_okikae_moji_0 = sbox[okikae_moji_0]   # w3_0の要素(8bit)をsboxで置換した文字列

# w3_1について
okikae_1 = []
for i in range(104, 112):
    okikae_1.append(binmoji_shift[i])
okikae_moji_1 = ''.join(okikae_1)
after_okikae_moji_1 = sbox[okikae_moji_1]

# w3_2について
okikae_2 = []
for i in range(112, 120):
    okikae_2.append(binmoji_shift[i])
okikae_moji_2 = ''.join(okikae_2)
after_okikae_moji_2 = sbox[okikae_moji_2]

# w3_3について
okikae_3 = []
for i in range(120, 128):
    okikae_3.append(binmoji_shift[i])
okikae_moji_3 = ''.join(okikae_3)
after_okikae_moji_3 = sbox[okikae_moji_3]

# 置き換え後ののw3を文字列として結合(32bit)
after_okikae_moji = after_okikae_moji_0 + after_okikae_moji_1 + after_okikae_moji_2 + after_okikae_moji_3

# 置き換え後の128bit全体
after_okikae_128 = []
for i in range(0, 96):     # w0～w2は変更なしなので、そのまま取得
    after_okikae_128.append(binmoji_shift_list[i])
for i in range(0, 32):     # w3は、after_okikae_mojiを取得
    after_okikae_128.append(after_okikae_moji[i])

after_okikae_128_moji = ''.join(after_okikae_128)

# w0の取得
emp_list = []
for i in range(0, 32):
    emp_list.append(after_okikae_128_moji[i])
w0 = ''.join(emp_list)
# w1の取得
emp_list = []
for i in range(32, 64):
    emp_list.append(after_okikae_128_moji[i])
w1 = ''.join(emp_list)
# w2の取得
emp_list = []
for i in range(64, 96):
    emp_list.append(after_okikae_128_moji[i])
w2 = ''.join(emp_list)
# w3の取得
emp_list = []
for i in range(96, 128):
    emp_list.append(after_okikae_128_moji[i])
w3 = ''.join(emp_list)

# 排他的論理和
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_10, after_okikae_moji)]
xor_1_new = []
for i in range(32):
    xor_1_new.append(str(xor_1[i]))
temp = ''.join(xor_1_new)

# ステップ4c(テキスト)
# j=0
w4_list = [ord(a) ^ ord(b) for a,b in zip(w0, temp)]
w4_list_str = []
for i in range(32):
    w4_list_str.append(str(w4_list[i]))
w4 = ''.join(w4_list_str)
temp = w4

# j=1
w5_list = [ord(a) ^ ord(b) for a,b in zip(w1, temp)]
w5_list_str = []
for i in range(32):
    w5_list_str.append(str(w5_list[i]))
w5 = ''.join(w5_list_str)
temp = w5

# j=2
w6_list = [ord(a) ^ ord(b) for a,b in zip(w2, temp)]
w6_list_str = []
for i in range(32):
    w6_list_str.append(str(w6_list[i]))
w6 = ''.join(w6_list_str)
temp = w6

# j=3
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3_mae, temp)]
w7_list_str = []
for i in range(32):
    w7_list_str.append(str(w7_list[i]))
w7 = ''.join(w7_list_str)
temp = w7

k10 = w4 + w5 + w6 + w7
binmoji = k10     # binmojiの更新

print('10個目のサブ鍵 k10')
print(k10)

k10_subkey = k10    # 後で変数名が被らないようにするため

print('サブ鍵生成を終了します。')
print('')



###################################
# データ暗号化アルゴリズム
###################################

# 状態行列の各要素の定義
emp_list = []
for i in range(0, 8):
    emp_list.append(hirabun[i])
s00 = ''.join(emp_list)

emp_list = []
for i in range(8, 16):
    emp_list.append(hirabun[i])
s10 = ''.join(emp_list)

emp_list = []
for i in range(16, 24):
    emp_list.append(hirabun[i])
s20 = ''.join(emp_list)

emp_list = []
for i in range(24, 32):
    emp_list.append(hirabun[i])
s30 = ''.join(emp_list)

emp_list = []
for i in range(32, 40):
    emp_list.append(hirabun[i])
s01 = ''.join(emp_list)

emp_list = []
for i in range(40, 48):
    emp_list.append(hirabun[i])
s11 = ''.join(emp_list)

emp_list = []
for i in range(48, 56):
    emp_list.append(hirabun[i])
s21 = ''.join(emp_list)

emp_list = []
for i in range(56, 64):
    emp_list.append(hirabun[i])
s31 = ''.join(emp_list)

emp_list = []
for i in range(64, 72):
    emp_list.append(hirabun[i])
s02 = ''.join(emp_list)

emp_list = []
for i in range(72, 80):
    emp_list.append(hirabun[i])
s12 = ''.join(emp_list)

emp_list = []
for i in range(80, 88):
    emp_list.append(hirabun[i])
s22 = ''.join(emp_list)

emp_list = []
for i in range(88, 96):
    emp_list.append(hirabun[i])
s32 = ''.join(emp_list)

emp_list = []
for i in range(96, 104):
    emp_list.append(hirabun[i])
s03 = ''.join(emp_list)

emp_list = []
for i in range(104, 112):
    emp_list.append(hirabun[i])
s13 = ''.join(emp_list)

emp_list = []
for i in range(112, 120):
    emp_list.append(hirabun[i])
s23 = ''.join(emp_list)

emp_list = []
for i in range(120, 128):
    emp_list.append(hirabun[i])
s33 = ''.join(emp_list)

# k0の行列の各要素の定義 
emp_list = []
for i in range(0, 8):
    emp_list.append(k0[i])
k00 = ''.join(emp_list)

emp_list = []
for i in range(8, 16):
    emp_list.append(k0[i])
k10 = ''.join(emp_list)

emp_list = []
for i in range(16, 24):
    emp_list.append(k0[i])
k20 = ''.join(emp_list)

emp_list = []
for i in range(24, 32):
    emp_list.append(k0[i])
k30 = ''.join(emp_list)

emp_list = []
for i in range(32, 40):
    emp_list.append(k0[i])
k01 = ''.join(emp_list)

emp_list = []
for i in range(40, 48):
    emp_list.append(k0[i])
k11 = ''.join(emp_list)

emp_list = []
for i in range(48, 56):
    emp_list.append(k0[i])
k21 = ''.join(emp_list)

emp_list = []
for i in range(56, 64):
    emp_list.append(k0[i])
k31 = ''.join(emp_list)

emp_list = []
for i in range(64, 72):
    emp_list.append(k0[i])
k02 = ''.join(emp_list)

emp_list = []
for i in range(72, 80):
    emp_list.append(k0[i])
k12 = ''.join(emp_list)

emp_list = []
for i in range(80, 88):
    emp_list.append(k0[i])
k22 = ''.join(emp_list)

emp_list = []
for i in range(88, 96):
    emp_list.append(k0[i])
k32 = ''.join(emp_list)

emp_list = []
for i in range(96, 104):
    emp_list.append(k0[i])
k03 = ''.join(emp_list)

emp_list = []
for i in range(104, 112):
    emp_list.append(k0[i])
k13 = ''.join(emp_list)

emp_list = []
for i in range(112, 120):
    emp_list.append(k0[i])
k23 = ''.join(emp_list)

emp_list = []
for i in range(120, 128):
    emp_list.append(k0[i])
k33 = ''.join(emp_list)


######### AddRoundKey(s, k0) ########
xor = [ord(a) ^ ord(b) for a,b in zip(s00, k00)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s00 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s10, k10)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s10 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s20, k20)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s20 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s30, k30)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s30 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s01, k01)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s01 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s11, k11)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s11 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s21, k21)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s21 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s31, k31)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s31 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s02, k02)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s02 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s12, k12)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s12 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s22, k22)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s22 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s32, k32)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s32 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s03, k03)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s03 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s13, k13)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s13 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s23, k23)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s23 = ''.join(xor_new)

xor = [ord(a) ^ ord(b) for a,b in zip(s33, k33)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s33 = ''.join(xor_new)


# ステップ4(テキスト)

####################
# i = 1
####################
# SubBytes(s)
s00 = sbox[s00]
s10 = sbox[s10]
s20 = sbox[s20]
s30 = sbox[s30]
s01 = sbox[s01]
s11 = sbox[s11]
s21 = sbox[s21]
s31 = sbox[s31]
s02 = sbox[s02]
s12 = sbox[s12]
s22 = sbox[s22]
s32 = sbox[s32]
s03 = sbox[s03]
s13 = sbox[s13]
s23 = sbox[s23]
s33 = sbox[s33]

# round[ 1].s_boxの確認
r1sbox = s00+s10+s20+s30+s01+s11+s21+s31+s02+s12+s22+s32+s03+s13+s23+s33
print('')
print('round[ 1].s_box')
print(r1sbox)


# ShiftRows(s)
s00_sr = s00
s01_sr = s01
s02_sr = s02
s03_sr = s03
s10_sr = s11
s11_sr = s12
s12_sr = s13
s13_sr = s10
s20_sr = s22
s21_sr = s23
s22_sr = s20
s23_sr = s21
s30_sr = s33
s31_sr = s30
s32_sr = s31
s33_sr = s32

# round[ 1].s_rowの確認
r1srow = s00_sr+s10_sr+s20_sr+s30_sr+s01_sr+s11_sr+s21_sr+s31_sr+s02_sr+s12_sr+s22_sr+s32_sr+s03_sr+s13_sr+s23_sr+s33_sr
print('round[ 1].s_row')
print(r1srow)

# MixColumns(s)
# 元のsの各要素を10進数に戻す
s00_sr_10 = (int(s00_sr[0]))*2**7 + (int(s00_sr[1]))*2**6 + (int(s00_sr[2]))*2**5 + (int(s00_sr[3]))*2**4 + (int(s00_sr[4]))*2**3 + (int(s00_sr[5]))*2**2 + (int(s00_sr[6]))*2**1 + (int(s00_sr[7]))*2**0
s01_sr_10 = (int(s01_sr[0]))*2**7 + (int(s01_sr[1]))*2**6 + (int(s01_sr[2]))*2**5 + (int(s01_sr[3]))*2**4 + (int(s01_sr[4]))*2**3 + (int(s01_sr[5]))*2**2 + (int(s01_sr[6]))*2**1 + (int(s01_sr[7]))*2**0
s02_sr_10 = (int(s02_sr[0]))*2**7 + (int(s02_sr[1]))*2**6 + (int(s02_sr[2]))*2**5 + (int(s02_sr[3]))*2**4 + (int(s02_sr[4]))*2**3 + (int(s02_sr[5]))*2**2 + (int(s02_sr[6]))*2**1 + (int(s02_sr[7]))*2**0
s03_sr_10 = (int(s03_sr[0]))*2**7 + (int(s03_sr[1]))*2**6 + (int(s03_sr[2]))*2**5 + (int(s03_sr[3]))*2**4 + (int(s03_sr[4]))*2**3 + (int(s03_sr[5]))*2**2 + (int(s03_sr[6]))*2**1 + (int(s03_sr[7]))*2**0
s10_sr_10 = (int(s10_sr[0]))*2**7 + (int(s10_sr[1]))*2**6 + (int(s10_sr[2]))*2**5 + (int(s10_sr[3]))*2**4 + (int(s10_sr[4]))*2**3 + (int(s10_sr[5]))*2**2 + (int(s10_sr[6]))*2**1 + (int(s10_sr[7]))*2**0
s11_sr_10 = (int(s11_sr[0]))*2**7 + (int(s11_sr[1]))*2**6 + (int(s11_sr[2]))*2**5 + (int(s11_sr[3]))*2**4 + (int(s11_sr[4]))*2**3 + (int(s11_sr[5]))*2**2 + (int(s11_sr[6]))*2**1 + (int(s11_sr[7]))*2**0
s12_sr_10 = (int(s12_sr[0]))*2**7 + (int(s12_sr[1]))*2**6 + (int(s12_sr[2]))*2**5 + (int(s12_sr[3]))*2**4 + (int(s12_sr[4]))*2**3 + (int(s12_sr[5]))*2**2 + (int(s12_sr[6]))*2**1 + (int(s12_sr[7]))*2**0
s13_sr_10 = (int(s13_sr[0]))*2**7 + (int(s13_sr[1]))*2**6 + (int(s13_sr[2]))*2**5 + (int(s13_sr[3]))*2**4 + (int(s13_sr[4]))*2**3 + (int(s13_sr[5]))*2**2 + (int(s13_sr[6]))*2**1 + (int(s13_sr[7]))*2**0
s20_sr_10 = (int(s20_sr[0]))*2**7 + (int(s20_sr[1]))*2**6 + (int(s20_sr[2]))*2**5 + (int(s20_sr[3]))*2**4 + (int(s20_sr[4]))*2**3 + (int(s20_sr[5]))*2**2 + (int(s20_sr[6]))*2**1 + (int(s20_sr[7]))*2**0
s21_sr_10 = (int(s21_sr[0]))*2**7 + (int(s21_sr[1]))*2**6 + (int(s21_sr[2]))*2**5 + (int(s21_sr[3]))*2**4 + (int(s21_sr[4]))*2**3 + (int(s21_sr[5]))*2**2 + (int(s21_sr[6]))*2**1 + (int(s21_sr[7]))*2**0
s22_sr_10 = (int(s22_sr[0]))*2**7 + (int(s22_sr[1]))*2**6 + (int(s22_sr[2]))*2**5 + (int(s22_sr[3]))*2**4 + (int(s22_sr[4]))*2**3 + (int(s22_sr[5]))*2**2 + (int(s22_sr[6]))*2**1 + (int(s22_sr[7]))*2**0
s23_sr_10 = (int(s23_sr[0]))*2**7 + (int(s23_sr[1]))*2**6 + (int(s23_sr[2]))*2**5 + (int(s23_sr[3]))*2**4 + (int(s23_sr[4]))*2**3 + (int(s23_sr[5]))*2**2 + (int(s23_sr[6]))*2**1 + (int(s23_sr[7]))*2**0
s30_sr_10 = (int(s30_sr[0]))*2**7 + (int(s30_sr[1]))*2**6 + (int(s30_sr[2]))*2**5 + (int(s30_sr[3]))*2**4 + (int(s30_sr[4]))*2**3 + (int(s30_sr[5]))*2**2 + (int(s30_sr[6]))*2**1 + (int(s30_sr[7]))*2**0
s31_sr_10 = (int(s31_sr[0]))*2**7 + (int(s31_sr[1]))*2**6 + (int(s31_sr[2]))*2**5 + (int(s31_sr[3]))*2**4 + (int(s31_sr[4]))*2**3 + (int(s31_sr[5]))*2**2 + (int(s31_sr[6]))*2**1 + (int(s31_sr[7]))*2**0
s32_sr_10 = (int(s32_sr[0]))*2**7 + (int(s32_sr[1]))*2**6 + (int(s32_sr[2]))*2**5 + (int(s32_sr[3]))*2**4 + (int(s32_sr[4]))*2**3 + (int(s32_sr[5]))*2**2 + (int(s32_sr[6]))*2**1 + (int(s32_sr[7]))*2**0
s33_sr_10 = (int(s33_sr[0]))*2**7 + (int(s33_sr[1]))*2**6 + (int(s33_sr[2]))*2**5 + (int(s33_sr[3]))*2**4 + (int(s33_sr[4]))*2**3 + (int(s33_sr[5]))*2**2 + (int(s33_sr[6]))*2**1 + (int(s33_sr[7]))*2**0
# これを基に、MixColumns後の各要素(例外処理を除く)を計算する
s00 = 2*(s00_sr_10) + 3*(s10_sr_10) + 1*(s20_sr_10) + 1*(s30_sr_10)
s01 = 2*(s01_sr_10) + 3*(s11_sr_10) + 1*(s21_sr_10) + 1*(s31_sr_10)
s02 = 2*(s02_sr_10) + 3*(s12_sr_10) + 1*(s22_sr_10) + 1*(s32_sr_10)
s03 = 2*(s03_sr_10) + 3*(s13_sr_10) + 1*(s23_sr_10) + 1*(s33_sr_10)
s10 = 1*(s00_sr_10) + 2*(s10_sr_10) + 3*(s20_sr_10) + 1*(s30_sr_10)
s11 = 1*(s01_sr_10) + 2*(s11_sr_10) + 3*(s21_sr_10) + 1*(s31_sr_10)
s12 = 1*(s02_sr_10) + 2*(s12_sr_10) + 3*(s22_sr_10) + 1*(s33_sr_10)
s13 = 1*(s03_sr_10) + 2*(s13_sr_10) + 3*(s23_sr_10) + 1*(s33_sr_10)
s20 = 1*(s00_sr_10) + 1*(s10_sr_10) + 2*(s20_sr_10) + 3*(s30_sr_10)
s21 = 1*(s01_sr_10) + 1*(s11_sr_10) + 2*(s21_sr_10) + 3*(s31_sr_10)
s22 = 1*(s02_sr_10) + 1*(s12_sr_10) + 2*(s22_sr_10) + 3*(s32_sr_10)
s23 = 1*(s03_sr_10) + 1*(s13_sr_10) + 2*(s23_sr_10) + 3*(s33_sr_10)
s30 = 3*(s00_sr_10) + 1*(s10_sr_10) + 1*(s20_sr_10) + 2*(s30_sr_10)
s31 = 3*(s01_sr_10) + 1*(s11_sr_10) + 1*(s21_sr_10) + 2*(s31_sr_10)
s32 = 3*(s02_sr_10) + 1*(s12_sr_10) + 1*(s22_sr_10) + 2*(s32_sr_10)
s33 = 3*(s03_sr_10) + 1*(s13_sr_10) + 1*(s23_sr_10) + 2*(s33_sr_10)
# 0bを付けない形式の2進数に直す
s00 = format(s00, 'b')
s01 = format(s01, 'b')
s02 = format(s02, 'b')
s03 = format(s03, 'b')
s10 = format(s10, 'b')
s11 = format(s11, 'b')
s12 = format(s12, 'b')
s13 = format(s13, 'b')
s20 = format(s20, 'b')
s21 = format(s21, 'b')
s22 = format(s22, 'b')
s23 = format(s23, 'b')
s30 = format(s30, 'b')
s31 = format(s31, 'b')
s32 = format(s32, 'b')
s33 = format(s33, 'b')
# 8桁に満たない場合、先頭に0を付けて8桁に揃える
if (len(s00) < 8):
    for i in range(8-len(s00)):
        s00 = '0' + s00
if (len(s01) < 8):
    for i in range(8-len(s01)):
        s01 = '0' + s01     
if (len(s02) < 8):
    for i in range(8-len(s02)):
        s02 = '0' + s02
if (len(s03) < 8):
    for i in range(8-len(s03)):
        s03 = '0' + s03
if (len(s10) < 8):
    for i in range(8-len(s10)):
        s10 = '0' + s10
if (len(s11) < 8):
    for i in range(8-len(s11)):
        s11 = '0' + s11
if (len(s12) < 8):
    for i in range(8-len(s12)):
        s12 = '0' + s12
if (len(s13) < 8):
    for i in range(8-len(s13)):
        s13 = '0' + s13
if (len(s20) < 8):
    for i in range(8-len(s20)):
        s20 = '0' + s20
if (len(s21) < 8):
    for i in range(8-len(s21)):
        s21 = '0' + s21
if (len(s22) < 8):
    for i in range(8-len(s22)):
        s22 = '0' + s22
if (len(s23) < 8):
    for i in range(8-len(s23)):
        s23 = '0' + s23
if (len(s30) < 8):
    for i in range(8-len(s30)):
        s30 = '0' + s30
if (len(s31) < 8):
    for i in range(8-len(s31)):
        s31 = '0' + s31
if (len(s32) < 8):
    for i in range(8-len(s32)):
        s32 = '0' + s32
if (len(s33) < 8):
    for i in range(8-len(s33)):
        s33 = '0' + s33

# 確認用
print('例外処理前のs01')
print(s01)
# 例外処理
##########
# s00
##########
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]

##########
# s01
##########
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]

# 確認用
print('例外処理後のs01')
print(s01)

##########
# s02
##########
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]

##########
# s03
##########
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]

##########
# s10
##########
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]

##########
# s11
##########
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]

##########
# s12
##########
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]

##########
# s13
##########
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []                 
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]

##########
# s20
##########
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]

##########
# s21
##########
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]

##########
# s22
##########
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]

##########
# s23
##########
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]

##########
# s30
##########
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]

##########
# s31
##########
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]

##########
# s32
##########
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]

##########
# s33
##########
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
# これにてMixColumnsが終了

# round[ 1].m_colの確認
r1mcol = s00+s10+s20+s30+s01+s11+s21+s31+s02+s12+s22+s32+s03+s13+s23+s33
print('round[ 1].mcol')
print(r1mcol)

# AddRoundKey(s, ki)
# k1行列の各要素を取得
emp_list = []
for i in range(0, 8):
    emp_list.append(k1[i])
k00 = ''.join(emp_list)
emp_list = []
for i in range(8, 16):
    emp_list.append(k1[i])
k10 = ''.join(emp_list)
emp_list = []
for i in range(16, 24):
    emp_list.append(k1[i])
k20 = ''.join(emp_list)
emp_list = []
for i in range(24, 32):
    emp_list.append(k1[i])
k30 = ''.join(emp_list)
emp_list = []
for i in range(32, 40):
    emp_list.append(k1[i])
k01 = ''.join(emp_list)
emp_list = []
for i in range(40, 48):
    emp_list.append(k1[i])
k11 = ''.join(emp_list)
emp_list = []
for i in range(48, 56):
    emp_list.append(k1[i])
k21 = ''.join(emp_list)
emp_list = []
for i in range(56, 64):
    emp_list.append(k1[i])
k31 = ''.join(emp_list)
emp_list = []
for i in range(64, 72):
    emp_list.append(k1[i])
k02 = ''.join(emp_list)
emp_list = []
for i in range(72, 80):
    emp_list.append(k1[i])
k12 = ''.join(emp_list)
emp_list = []
for i in range(80, 88):
    emp_list.append(k1[i])
k22 = ''.join(emp_list)
emp_list = []
for i in range(88, 96):
    emp_list.append(k1[i])
k32 = ''.join(emp_list)
emp_list = []
for i in range(96, 104):
    emp_list.append(k1[i])
k03 = ''.join(emp_list)
emp_list = []
for i in range(104, 112):
    emp_list.append(k1[i])
k13 = ''.join(emp_list)
emp_list = []
for i in range(112, 120):
    emp_list.append(k1[i])
k23 = ''.join(emp_list)
emp_list = []
for i in range(120, 128):
    emp_list.append(k1[i])
k33 = ''.join(emp_list)

# 対応する要素ごとに排他的論理和を取る
xor = [ord(a) ^ ord(b) for a,b in zip(s00, k00)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s00 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s10, k10)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s10 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s20, k20)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s20 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s30, k30)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s30 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s01, k01)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s01 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s11, k11)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s11 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s21, k21)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s21 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s31, k31)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s31 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s02, k02)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s02 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s12, k12)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s12 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s22, k22)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s22 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s32, k32)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s32 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s03, k03)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s03 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s13, k13)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s13 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s23, k23)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s23 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s33, k33)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s33 = ''.join(xor_new)


####################
# i = 2
####################
# SubBytes(s)
s00 = sbox[s00]
s10 = sbox[s10]
s20 = sbox[s20]
s30 = sbox[s30]
s01 = sbox[s01]
s11 = sbox[s11]
s21 = sbox[s21]
s31 = sbox[s31]
s02 = sbox[s02]
s12 = sbox[s12]
s22 = sbox[s22]
s32 = sbox[s32]
s03 = sbox[s03]
s13 = sbox[s13]
s23 = sbox[s23]
s33 = sbox[s33]

# ShiftRows(s)
s00_sr = s00
s01_sr = s01
s02_sr = s02
s03_sr = s03
s10_sr = s11
s11_sr = s12
s12_sr = s13
s13_sr = s10
s20_sr = s22
s21_sr = s23
s22_sr = s20
s23_sr = s21
s30_sr = s33
s31_sr = s30
s32_sr = s31
s33_sr = s32

# MixColumns(s)
# 元のsの各要素を10進数に戻す
s00_sr_10 = (int(s00_sr[0]))*2**7 + (int(s00_sr[1]))*2**6 + (int(s00_sr[2]))*2**5 + (int(s00_sr[3]))*2**4 + (int(s00_sr[4]))*2**3 + (int(s00_sr[5]))*2**2 + (int(s00_sr[6]))*2**1 + (int(s00_sr[7]))*2**0
s01_sr_10 = (int(s01_sr[0]))*2**7 + (int(s01_sr[1]))*2**6 + (int(s01_sr[2]))*2**5 + (int(s01_sr[3]))*2**4 + (int(s01_sr[4]))*2**3 + (int(s01_sr[5]))*2**2 + (int(s01_sr[6]))*2**1 + (int(s01_sr[7]))*2**0
s02_sr_10 = (int(s02_sr[0]))*2**7 + (int(s02_sr[1]))*2**6 + (int(s02_sr[2]))*2**5 + (int(s02_sr[3]))*2**4 + (int(s02_sr[4]))*2**3 + (int(s02_sr[5]))*2**2 + (int(s02_sr[6]))*2**1 + (int(s02_sr[7]))*2**0
s03_sr_10 = (int(s03_sr[0]))*2**7 + (int(s03_sr[1]))*2**6 + (int(s03_sr[2]))*2**5 + (int(s03_sr[3]))*2**4 + (int(s03_sr[4]))*2**3 + (int(s03_sr[5]))*2**2 + (int(s03_sr[6]))*2**1 + (int(s03_sr[7]))*2**0
s10_sr_10 = (int(s10_sr[0]))*2**7 + (int(s10_sr[1]))*2**6 + (int(s10_sr[2]))*2**5 + (int(s10_sr[3]))*2**4 + (int(s10_sr[4]))*2**3 + (int(s10_sr[5]))*2**2 + (int(s10_sr[6]))*2**1 + (int(s10_sr[7]))*2**0
s11_sr_10 = (int(s11_sr[0]))*2**7 + (int(s11_sr[1]))*2**6 + (int(s11_sr[2]))*2**5 + (int(s11_sr[3]))*2**4 + (int(s11_sr[4]))*2**3 + (int(s11_sr[5]))*2**2 + (int(s11_sr[6]))*2**1 + (int(s11_sr[7]))*2**0
s12_sr_10 = (int(s12_sr[0]))*2**7 + (int(s12_sr[1]))*2**6 + (int(s12_sr[2]))*2**5 + (int(s12_sr[3]))*2**4 + (int(s12_sr[4]))*2**3 + (int(s12_sr[5]))*2**2 + (int(s12_sr[6]))*2**1 + (int(s12_sr[7]))*2**0
s13_sr_10 = (int(s13_sr[0]))*2**7 + (int(s13_sr[1]))*2**6 + (int(s13_sr[2]))*2**5 + (int(s13_sr[3]))*2**4 + (int(s13_sr[4]))*2**3 + (int(s13_sr[5]))*2**2 + (int(s13_sr[6]))*2**1 + (int(s13_sr[7]))*2**0
s20_sr_10 = (int(s20_sr[0]))*2**7 + (int(s20_sr[1]))*2**6 + (int(s20_sr[2]))*2**5 + (int(s20_sr[3]))*2**4 + (int(s20_sr[4]))*2**3 + (int(s20_sr[5]))*2**2 + (int(s20_sr[6]))*2**1 + (int(s20_sr[7]))*2**0
s21_sr_10 = (int(s21_sr[0]))*2**7 + (int(s21_sr[1]))*2**6 + (int(s21_sr[2]))*2**5 + (int(s21_sr[3]))*2**4 + (int(s21_sr[4]))*2**3 + (int(s21_sr[5]))*2**2 + (int(s21_sr[6]))*2**1 + (int(s21_sr[7]))*2**0
s22_sr_10 = (int(s22_sr[0]))*2**7 + (int(s22_sr[1]))*2**6 + (int(s22_sr[2]))*2**5 + (int(s22_sr[3]))*2**4 + (int(s22_sr[4]))*2**3 + (int(s22_sr[5]))*2**2 + (int(s22_sr[6]))*2**1 + (int(s22_sr[7]))*2**0
s23_sr_10 = (int(s23_sr[0]))*2**7 + (int(s23_sr[1]))*2**6 + (int(s23_sr[2]))*2**5 + (int(s23_sr[3]))*2**4 + (int(s23_sr[4]))*2**3 + (int(s23_sr[5]))*2**2 + (int(s23_sr[6]))*2**1 + (int(s23_sr[7]))*2**0
s30_sr_10 = (int(s30_sr[0]))*2**7 + (int(s30_sr[1]))*2**6 + (int(s30_sr[2]))*2**5 + (int(s30_sr[3]))*2**4 + (int(s30_sr[4]))*2**3 + (int(s30_sr[5]))*2**2 + (int(s30_sr[6]))*2**1 + (int(s30_sr[7]))*2**0
s31_sr_10 = (int(s31_sr[0]))*2**7 + (int(s31_sr[1]))*2**6 + (int(s31_sr[2]))*2**5 + (int(s31_sr[3]))*2**4 + (int(s31_sr[4]))*2**3 + (int(s31_sr[5]))*2**2 + (int(s31_sr[6]))*2**1 + (int(s31_sr[7]))*2**0
s32_sr_10 = (int(s32_sr[0]))*2**7 + (int(s32_sr[1]))*2**6 + (int(s32_sr[2]))*2**5 + (int(s32_sr[3]))*2**4 + (int(s32_sr[4]))*2**3 + (int(s32_sr[5]))*2**2 + (int(s32_sr[6]))*2**1 + (int(s32_sr[7]))*2**0
s33_sr_10 = (int(s33_sr[0]))*2**7 + (int(s33_sr[1]))*2**6 + (int(s33_sr[2]))*2**5 + (int(s33_sr[3]))*2**4 + (int(s33_sr[4]))*2**3 + (int(s33_sr[5]))*2**2 + (int(s33_sr[6]))*2**1 + (int(s33_sr[7]))*2**0
# これを基に、MixColumns後の各要素(例外処理を除く)を計算する
s00 = 2*(s00_sr_10) + 3*(s10_sr_10) + 1*(s20_sr_10) + 1*(s30_sr_10)
s01 = 2*(s01_sr_10) + 3*(s11_sr_10) + 1*(s21_sr_10) + 1*(s31_sr_10)
s02 = 2*(s02_sr_10) + 3*(s12_sr_10) + 1*(s22_sr_10) + 1*(s32_sr_10)
s03 = 2*(s03_sr_10) + 3*(s13_sr_10) + 1*(s23_sr_10) + 1*(s33_sr_10)
s10 = 1*(s00_sr_10) + 2*(s10_sr_10) + 3*(s20_sr_10) + 1*(s30_sr_10)
s11 = 1*(s01_sr_10) + 2*(s11_sr_10) + 3*(s21_sr_10) + 1*(s31_sr_10)
s12 = 1*(s02_sr_10) + 2*(s12_sr_10) + 3*(s22_sr_10) + 1*(s33_sr_10)
s13 = 1*(s03_sr_10) + 2*(s13_sr_10) + 3*(s23_sr_10) + 1*(s33_sr_10)
s20 = 1*(s00_sr_10) + 1*(s10_sr_10) + 2*(s20_sr_10) + 3*(s30_sr_10)
s21 = 1*(s01_sr_10) + 1*(s11_sr_10) + 2*(s21_sr_10) + 3*(s31_sr_10)
s22 = 1*(s02_sr_10) + 1*(s12_sr_10) + 2*(s22_sr_10) + 3*(s32_sr_10)
s23 = 1*(s03_sr_10) + 1*(s13_sr_10) + 2*(s23_sr_10) + 3*(s33_sr_10)
s30 = 3*(s00_sr_10) + 1*(s10_sr_10) + 1*(s20_sr_10) + 2*(s30_sr_10)
s31 = 3*(s01_sr_10) + 1*(s11_sr_10) + 1*(s21_sr_10) + 2*(s31_sr_10)
s32 = 3*(s02_sr_10) + 1*(s12_sr_10) + 1*(s22_sr_10) + 2*(s32_sr_10)
s33 = 3*(s03_sr_10) + 1*(s13_sr_10) + 1*(s23_sr_10) + 2*(s33_sr_10)
# 0bを付けない形式の2進数に直す
s00 = format(s00, 'b')
s01 = format(s01, 'b')
s02 = format(s02, 'b')
s03 = format(s03, 'b')
s10 = format(s10, 'b')
s11 = format(s11, 'b')
s12 = format(s12, 'b')
s13 = format(s13, 'b')
s20 = format(s20, 'b')
s21 = format(s21, 'b')
s22 = format(s22, 'b')
s23 = format(s23, 'b')
s30 = format(s30, 'b')
s31 = format(s31, 'b')
s32 = format(s32, 'b')
s33 = format(s33, 'b')
# 8桁に満たない場合、先頭に0を付けて8桁に揃える
if (len(s00) < 8):
    for i in range(8-len(s00)):
        s00 = '0' + s00
if (len(s01) < 8):
    for i in range(8-len(s01)):
        s01 = '0' + s01     
if (len(s02) < 8):
    for i in range(8-len(s02)):
        s02 = '0' + s02
if (len(s03) < 8):
    for i in range(8-len(s03)):
        s03 = '0' + s03
if (len(s10) < 8):
    for i in range(8-len(s10)):
        s10 = '0' + s10
if (len(s11) < 8):
    for i in range(8-len(s11)):
        s11 = '0' + s11
if (len(s12) < 8):
    for i in range(8-len(s12)):
        s12 = '0' + s12
if (len(s13) < 8):
    for i in range(8-len(s13)):
        s13 = '0' + s13
if (len(s20) < 8):
    for i in range(8-len(s20)):
        s20 = '0' + s20
if (len(s21) < 8):
    for i in range(8-len(s21)):
        s21 = '0' + s21
if (len(s22) < 8):
    for i in range(8-len(s22)):
        s22 = '0' + s22
if (len(s23) < 8):
    for i in range(8-len(s23)):
        s23 = '0' + s23
if (len(s30) < 8):
    for i in range(8-len(s30)):
        s30 = '0' + s30
if (len(s31) < 8):
    for i in range(8-len(s31)):
        s31 = '0' + s31
if (len(s32) < 8):
    for i in range(8-len(s32)):
        s32 = '0' + s32
if (len(s33) < 8):
    for i in range(8-len(s33)):
        s33 = '0' + s33
# 例外処理
##########
# s00
##########
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]

##########
# s01
##########
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]

##########
# s02
##########
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]

##########
# s03
##########
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]

##########
# s10
##########
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]

##########
# s11
##########
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]

##########
# s12
##########
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]

##########
# s13
##########
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]

##########
# s20
##########
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]

##########
# s21
##########
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]

##########
# s22
##########
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]

##########
# s23
##########
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]

##########
# s30
##########
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]

##########
# s31
##########
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]

##########
# s32
##########
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]

##########
# s33
##########
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
# これにてMixColumnsが終了

# AddRoundKey(s, ki)
# k2行列の各要素を取得
emp_list = []
for i in range(0, 8):
    emp_list.append(k2[i])
k00 = ''.join(emp_list)
emp_list = []
for i in range(8, 16):
    emp_list.append(k2[i])
k10 = ''.join(emp_list)
emp_list = []
for i in range(16, 24):
    emp_list.append(k2[i])
k20 = ''.join(emp_list)
emp_list = []
for i in range(24, 32):
    emp_list.append(k2[i])
k30 = ''.join(emp_list)
emp_list = []
for i in range(32, 40):
    emp_list.append(k2[i])
k01 = ''.join(emp_list)
emp_list = []
for i in range(40, 48):
    emp_list.append(k2[i])
k11 = ''.join(emp_list)
emp_list = []
for i in range(48, 56):
    emp_list.append(k2[i])
k21 = ''.join(emp_list)
emp_list = []
for i in range(56, 64):
    emp_list.append(k2[i])
k31 = ''.join(emp_list)
emp_list = []
for i in range(64, 72):
    emp_list.append(k2[i])
k02 = ''.join(emp_list)
emp_list = []
for i in range(72, 80):
    emp_list.append(k2[i])
k12 = ''.join(emp_list)
emp_list = []
for i in range(80, 88):
    emp_list.append(k2[i])
k22 = ''.join(emp_list)
emp_list = []
for i in range(88, 96):
    emp_list.append(k2[i])
k32 = ''.join(emp_list)
emp_list = []
for i in range(96, 104):
    emp_list.append(k2[i])
k03 = ''.join(emp_list)
emp_list = []
for i in range(104, 112):
    emp_list.append(k2[i])
k13 = ''.join(emp_list)
emp_list = []
for i in range(112, 120):
    emp_list.append(k2[i])
k23 = ''.join(emp_list)
emp_list = []
for i in range(120, 128):
    emp_list.append(k2[i])
k33 = ''.join(emp_list)

# 対応する要素ごとに排他的論理和を取る
xor = [ord(a) ^ ord(b) for a,b in zip(s00, k00)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s00 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s10, k10)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s10 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s20, k20)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s20 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s30, k30)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s30 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s01, k01)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s01 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s11, k11)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s11 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s21, k21)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s21 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s31, k31)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s31 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s02, k02)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s02 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s12, k12)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s12 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s22, k22)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s22 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s32, k32)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s32 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s03, k03)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s03 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s13, k13)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s13 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s23, k23)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s23 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s33, k33)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s33 = ''.join(xor_new)


####################
# i = 3
####################
# SubBytes(s)
s00 = sbox[s00]
s10 = sbox[s10]
s20 = sbox[s20]
s30 = sbox[s30]
s01 = sbox[s01]
s11 = sbox[s11]
s21 = sbox[s21]
s31 = sbox[s31]
s02 = sbox[s02]
s12 = sbox[s12]
s22 = sbox[s22]
s32 = sbox[s32]
s03 = sbox[s03]
s13 = sbox[s13]
s23 = sbox[s23]
s33 = sbox[s33]

# ShiftRows(s)
s00_sr = s00
s01_sr = s01
s02_sr = s02
s03_sr = s03
s10_sr = s11
s11_sr = s12
s12_sr = s13
s13_sr = s10
s20_sr = s22
s21_sr = s23
s22_sr = s20
s23_sr = s21
s30_sr = s33
s31_sr = s30
s32_sr = s31
s33_sr = s32

# MixColumns(s)
# 元のsの各要素を10進数に戻す
s00_sr_10 = (int(s00_sr[0]))*2**7 + (int(s00_sr[1]))*2**6 + (int(s00_sr[2]))*2**5 + (int(s00_sr[3]))*2**4 + (int(s00_sr[4]))*2**3 + (int(s00_sr[5]))*2**2 + (int(s00_sr[6]))*2**1 + (int(s00_sr[7]))*2**0
s01_sr_10 = (int(s01_sr[0]))*2**7 + (int(s01_sr[1]))*2**6 + (int(s01_sr[2]))*2**5 + (int(s01_sr[3]))*2**4 + (int(s01_sr[4]))*2**3 + (int(s01_sr[5]))*2**2 + (int(s01_sr[6]))*2**1 + (int(s01_sr[7]))*2**0
s02_sr_10 = (int(s02_sr[0]))*2**7 + (int(s02_sr[1]))*2**6 + (int(s02_sr[2]))*2**5 + (int(s02_sr[3]))*2**4 + (int(s02_sr[4]))*2**3 + (int(s02_sr[5]))*2**2 + (int(s02_sr[6]))*2**1 + (int(s02_sr[7]))*2**0
s03_sr_10 = (int(s03_sr[0]))*2**7 + (int(s03_sr[1]))*2**6 + (int(s03_sr[2]))*2**5 + (int(s03_sr[3]))*2**4 + (int(s03_sr[4]))*2**3 + (int(s03_sr[5]))*2**2 + (int(s03_sr[6]))*2**1 + (int(s03_sr[7]))*2**0
s10_sr_10 = (int(s10_sr[0]))*2**7 + (int(s10_sr[1]))*2**6 + (int(s10_sr[2]))*2**5 + (int(s10_sr[3]))*2**4 + (int(s10_sr[4]))*2**3 + (int(s10_sr[5]))*2**2 + (int(s10_sr[6]))*2**1 + (int(s10_sr[7]))*2**0
s11_sr_10 = (int(s11_sr[0]))*2**7 + (int(s11_sr[1]))*2**6 + (int(s11_sr[2]))*2**5 + (int(s11_sr[3]))*2**4 + (int(s11_sr[4]))*2**3 + (int(s11_sr[5]))*2**2 + (int(s11_sr[6]))*2**1 + (int(s11_sr[7]))*2**0
s12_sr_10 = (int(s12_sr[0]))*2**7 + (int(s12_sr[1]))*2**6 + (int(s12_sr[2]))*2**5 + (int(s12_sr[3]))*2**4 + (int(s12_sr[4]))*2**3 + (int(s12_sr[5]))*2**2 + (int(s12_sr[6]))*2**1 + (int(s12_sr[7]))*2**0
s13_sr_10 = (int(s13_sr[0]))*2**7 + (int(s13_sr[1]))*2**6 + (int(s13_sr[2]))*2**5 + (int(s13_sr[3]))*2**4 + (int(s13_sr[4]))*2**3 + (int(s13_sr[5]))*2**2 + (int(s13_sr[6]))*2**1 + (int(s13_sr[7]))*2**0
s20_sr_10 = (int(s20_sr[0]))*2**7 + (int(s20_sr[1]))*2**6 + (int(s20_sr[2]))*2**5 + (int(s20_sr[3]))*2**4 + (int(s20_sr[4]))*2**3 + (int(s20_sr[5]))*2**2 + (int(s20_sr[6]))*2**1 + (int(s20_sr[7]))*2**0
s21_sr_10 = (int(s21_sr[0]))*2**7 + (int(s21_sr[1]))*2**6 + (int(s21_sr[2]))*2**5 + (int(s21_sr[3]))*2**4 + (int(s21_sr[4]))*2**3 + (int(s21_sr[5]))*2**2 + (int(s21_sr[6]))*2**1 + (int(s21_sr[7]))*2**0
s22_sr_10 = (int(s22_sr[0]))*2**7 + (int(s22_sr[1]))*2**6 + (int(s22_sr[2]))*2**5 + (int(s22_sr[3]))*2**4 + (int(s22_sr[4]))*2**3 + (int(s22_sr[5]))*2**2 + (int(s22_sr[6]))*2**1 + (int(s22_sr[7]))*2**0
s23_sr_10 = (int(s23_sr[0]))*2**7 + (int(s23_sr[1]))*2**6 + (int(s23_sr[2]))*2**5 + (int(s23_sr[3]))*2**4 + (int(s23_sr[4]))*2**3 + (int(s23_sr[5]))*2**2 + (int(s23_sr[6]))*2**1 + (int(s23_sr[7]))*2**0
s30_sr_10 = (int(s30_sr[0]))*2**7 + (int(s30_sr[1]))*2**6 + (int(s30_sr[2]))*2**5 + (int(s30_sr[3]))*2**4 + (int(s30_sr[4]))*2**3 + (int(s30_sr[5]))*2**2 + (int(s30_sr[6]))*2**1 + (int(s30_sr[7]))*2**0
s31_sr_10 = (int(s31_sr[0]))*2**7 + (int(s31_sr[1]))*2**6 + (int(s31_sr[2]))*2**5 + (int(s31_sr[3]))*2**4 + (int(s31_sr[4]))*2**3 + (int(s31_sr[5]))*2**2 + (int(s31_sr[6]))*2**1 + (int(s31_sr[7]))*2**0
s32_sr_10 = (int(s32_sr[0]))*2**7 + (int(s32_sr[1]))*2**6 + (int(s32_sr[2]))*2**5 + (int(s32_sr[3]))*2**4 + (int(s32_sr[4]))*2**3 + (int(s32_sr[5]))*2**2 + (int(s32_sr[6]))*2**1 + (int(s32_sr[7]))*2**0
s33_sr_10 = (int(s33_sr[0]))*2**7 + (int(s33_sr[1]))*2**6 + (int(s33_sr[2]))*2**5 + (int(s33_sr[3]))*2**4 + (int(s33_sr[4]))*2**3 + (int(s33_sr[5]))*2**2 + (int(s33_sr[6]))*2**1 + (int(s33_sr[7]))*2**0
# これを基に、MixColumns後の各要素(例外処理を除く)を計算する
s00 = 2*(s00_sr_10) + 3*(s10_sr_10) + 1*(s20_sr_10) + 1*(s30_sr_10)
s01 = 2*(s01_sr_10) + 3*(s11_sr_10) + 1*(s21_sr_10) + 1*(s31_sr_10)
s02 = 2*(s02_sr_10) + 3*(s12_sr_10) + 1*(s22_sr_10) + 1*(s32_sr_10)
s03 = 2*(s03_sr_10) + 3*(s13_sr_10) + 1*(s23_sr_10) + 1*(s33_sr_10)
s10 = 1*(s00_sr_10) + 2*(s10_sr_10) + 3*(s20_sr_10) + 1*(s30_sr_10)
s11 = 1*(s01_sr_10) + 2*(s11_sr_10) + 3*(s21_sr_10) + 1*(s31_sr_10)
s12 = 1*(s02_sr_10) + 2*(s12_sr_10) + 3*(s22_sr_10) + 1*(s33_sr_10)
s13 = 1*(s03_sr_10) + 2*(s13_sr_10) + 3*(s23_sr_10) + 1*(s33_sr_10)
s20 = 1*(s00_sr_10) + 1*(s10_sr_10) + 2*(s20_sr_10) + 3*(s30_sr_10)
s21 = 1*(s01_sr_10) + 1*(s11_sr_10) + 2*(s21_sr_10) + 3*(s31_sr_10)
s22 = 1*(s02_sr_10) + 1*(s12_sr_10) + 2*(s22_sr_10) + 3*(s32_sr_10)
s23 = 1*(s03_sr_10) + 1*(s13_sr_10) + 2*(s23_sr_10) + 3*(s33_sr_10)
s30 = 3*(s00_sr_10) + 1*(s10_sr_10) + 1*(s20_sr_10) + 2*(s30_sr_10)
s31 = 3*(s01_sr_10) + 1*(s11_sr_10) + 1*(s21_sr_10) + 2*(s31_sr_10)
s32 = 3*(s02_sr_10) + 1*(s12_sr_10) + 1*(s22_sr_10) + 2*(s32_sr_10)
s33 = 3*(s03_sr_10) + 1*(s13_sr_10) + 1*(s23_sr_10) + 2*(s33_sr_10)
# 0bを付けない形式の2進数に直す
s00 = format(s00, 'b')
s01 = format(s01, 'b')
s02 = format(s02, 'b')
s03 = format(s03, 'b')
s10 = format(s10, 'b')
s11 = format(s11, 'b')
s12 = format(s12, 'b')
s13 = format(s13, 'b')
s20 = format(s20, 'b')
s21 = format(s21, 'b')
s22 = format(s22, 'b')
s23 = format(s23, 'b')
s30 = format(s30, 'b')
s31 = format(s31, 'b')
s32 = format(s32, 'b')
s33 = format(s33, 'b')
# 8桁に満たない場合、先頭に0を付けて8桁に揃える
if (len(s00) < 8):
    for i in range(8-len(s00)):
        s00 = '0' + s00
if (len(s01) < 8):
    for i in range(8-len(s01)):
        s01 = '0' + s01     
if (len(s02) < 8):
    for i in range(8-len(s02)):
        s02 = '0' + s02
if (len(s03) < 8):
    for i in range(8-len(s03)):
        s03 = '0' + s03
if (len(s10) < 8):
    for i in range(8-len(s10)):
        s10 = '0' + s10
if (len(s11) < 8):
    for i in range(8-len(s11)):
        s11 = '0' + s11
if (len(s12) < 8):
    for i in range(8-len(s12)):
        s12 = '0' + s12
if (len(s13) < 8):
    for i in range(8-len(s13)):
        s13 = '0' + s13
if (len(s20) < 8):
    for i in range(8-len(s20)):
        s20 = '0' + s20
if (len(s21) < 8):
    for i in range(8-len(s21)):
        s21 = '0' + s21
if (len(s22) < 8):
    for i in range(8-len(s22)):
        s22 = '0' + s22
if (len(s23) < 8):
    for i in range(8-len(s23)):
        s23 = '0' + s23
if (len(s30) < 8):
    for i in range(8-len(s30)):
        s30 = '0' + s30
if (len(s31) < 8):
    for i in range(8-len(s31)):
        s31 = '0' + s31
if (len(s32) < 8):
    for i in range(8-len(s32)):
        s32 = '0' + s32
if (len(s33) < 8):
    for i in range(8-len(s33)):
        s33 = '0' + s33
# 例外処理
##########
# s00
##########
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]

##########
# s01
##########
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]

##########
# s02
##########
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]

##########
# s03
##########
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]

##########
# s10
##########
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]

##########
# s11
##########
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]

##########
# s12
##########
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]

##########
# s13
##########
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]

##########
# s20
##########
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]

##########
# s21
##########
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]

##########
# s22
##########
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]

##########
# s23
##########
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]

##########
# s30
##########
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]

##########
# s31
##########
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]

##########
# s32
##########
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]

##########
# s33
##########
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
# これにてMixColumnsが終了

# AddRoundKey(s, ki)
# k3行列の各要素を取得
emp_list = []
for i in range(0, 8):
    emp_list.append(k3[i])
k00 = ''.join(emp_list)
emp_list = []
for i in range(8, 16):
    emp_list.append(k3[i])
k10 = ''.join(emp_list)
emp_list = []
for i in range(16, 24):
    emp_list.append(k3[i])
k20 = ''.join(emp_list)
emp_list = []
for i in range(24, 32):
    emp_list.append(k3[i])
k30 = ''.join(emp_list)
emp_list = []
for i in range(32, 40):
    emp_list.append(k3[i])
k01 = ''.join(emp_list)
emp_list = []
for i in range(40, 48):
    emp_list.append(k3[i])
k11 = ''.join(emp_list)
emp_list = []
for i in range(48, 56):
    emp_list.append(k3[i])
k21 = ''.join(emp_list)
emp_list = []
for i in range(56, 64):
    emp_list.append(k3[i])
k31 = ''.join(emp_list)
emp_list = []
for i in range(64, 72):
    emp_list.append(k3[i])
k02 = ''.join(emp_list)
emp_list = []
for i in range(72, 80):
    emp_list.append(k3[i])
k12 = ''.join(emp_list)
emp_list = []
for i in range(80, 88):
    emp_list.append(k3[i])
k22 = ''.join(emp_list)
emp_list = []
for i in range(88, 96):
    emp_list.append(k3[i])
k32 = ''.join(emp_list)
emp_list = []
for i in range(96, 104):
    emp_list.append(k3[i])
k03 = ''.join(emp_list)
emp_list = []
for i in range(104, 112):
    emp_list.append(k3[i])
k13 = ''.join(emp_list)
emp_list = []
for i in range(112, 120):
    emp_list.append(k3[i])
k23 = ''.join(emp_list)
emp_list = []
for i in range(120, 128):
    emp_list.append(k3[i])
k33 = ''.join(emp_list)

# 対応する要素ごとに排他的論理和を取る
xor = [ord(a) ^ ord(b) for a,b in zip(s00, k00)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s00 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s10, k10)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s10 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s20, k20)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s20 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s30, k30)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s30 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s01, k01)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s01 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s11, k11)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s11 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s21, k21)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s21 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s31, k31)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s31 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s02, k02)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s02 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s12, k12)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s12 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s22, k22)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s22 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s32, k32)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s32 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s03, k03)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s03 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s13, k13)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s13 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s23, k23)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s23 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s33, k33)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s33 = ''.join(xor_new)


####################
# i = 4
####################
# SubBytes(s)
s00 = sbox[s00]
s10 = sbox[s10]
s20 = sbox[s20]
s30 = sbox[s30]
s01 = sbox[s01]
s11 = sbox[s11]
s21 = sbox[s21]
s31 = sbox[s31]
s02 = sbox[s02]
s12 = sbox[s12]
s22 = sbox[s22]
s32 = sbox[s32]
s03 = sbox[s03]
s13 = sbox[s13]
s23 = sbox[s23]
s33 = sbox[s33]

# ShiftRows(s)
s00_sr = s00
s01_sr = s01
s02_sr = s02
s03_sr = s03
s10_sr = s11
s11_sr = s12
s12_sr = s13
s13_sr = s10
s20_sr = s22
s21_sr = s23
s22_sr = s20
s23_sr = s21
s30_sr = s33
s31_sr = s30
s32_sr = s31
s33_sr = s32

# MixColumns(s)
# 元のsの各要素を10進数に戻す
s00_sr_10 = (int(s00_sr[0]))*2**7 + (int(s00_sr[1]))*2**6 + (int(s00_sr[2]))*2**5 + (int(s00_sr[3]))*2**4 + (int(s00_sr[4]))*2**3 + (int(s00_sr[5]))*2**2 + (int(s00_sr[6]))*2**1 + (int(s00_sr[7]))*2**0
s01_sr_10 = (int(s01_sr[0]))*2**7 + (int(s01_sr[1]))*2**6 + (int(s01_sr[2]))*2**5 + (int(s01_sr[3]))*2**4 + (int(s01_sr[4]))*2**3 + (int(s01_sr[5]))*2**2 + (int(s01_sr[6]))*2**1 + (int(s01_sr[7]))*2**0
s02_sr_10 = (int(s02_sr[0]))*2**7 + (int(s02_sr[1]))*2**6 + (int(s02_sr[2]))*2**5 + (int(s02_sr[3]))*2**4 + (int(s02_sr[4]))*2**3 + (int(s02_sr[5]))*2**2 + (int(s02_sr[6]))*2**1 + (int(s02_sr[7]))*2**0
s03_sr_10 = (int(s03_sr[0]))*2**7 + (int(s03_sr[1]))*2**6 + (int(s03_sr[2]))*2**5 + (int(s03_sr[3]))*2**4 + (int(s03_sr[4]))*2**3 + (int(s03_sr[5]))*2**2 + (int(s03_sr[6]))*2**1 + (int(s03_sr[7]))*2**0
s10_sr_10 = (int(s10_sr[0]))*2**7 + (int(s10_sr[1]))*2**6 + (int(s10_sr[2]))*2**5 + (int(s10_sr[3]))*2**4 + (int(s10_sr[4]))*2**3 + (int(s10_sr[5]))*2**2 + (int(s10_sr[6]))*2**1 + (int(s10_sr[7]))*2**0
s11_sr_10 = (int(s11_sr[0]))*2**7 + (int(s11_sr[1]))*2**6 + (int(s11_sr[2]))*2**5 + (int(s11_sr[3]))*2**4 + (int(s11_sr[4]))*2**3 + (int(s11_sr[5]))*2**2 + (int(s11_sr[6]))*2**1 + (int(s11_sr[7]))*2**0
s12_sr_10 = (int(s12_sr[0]))*2**7 + (int(s12_sr[1]))*2**6 + (int(s12_sr[2]))*2**5 + (int(s12_sr[3]))*2**4 + (int(s12_sr[4]))*2**3 + (int(s12_sr[5]))*2**2 + (int(s12_sr[6]))*2**1 + (int(s12_sr[7]))*2**0
s13_sr_10 = (int(s13_sr[0]))*2**7 + (int(s13_sr[1]))*2**6 + (int(s13_sr[2]))*2**5 + (int(s13_sr[3]))*2**4 + (int(s13_sr[4]))*2**3 + (int(s13_sr[5]))*2**2 + (int(s13_sr[6]))*2**1 + (int(s13_sr[7]))*2**0
s20_sr_10 = (int(s20_sr[0]))*2**7 + (int(s20_sr[1]))*2**6 + (int(s20_sr[2]))*2**5 + (int(s20_sr[3]))*2**4 + (int(s20_sr[4]))*2**3 + (int(s20_sr[5]))*2**2 + (int(s20_sr[6]))*2**1 + (int(s20_sr[7]))*2**0
s21_sr_10 = (int(s21_sr[0]))*2**7 + (int(s21_sr[1]))*2**6 + (int(s21_sr[2]))*2**5 + (int(s21_sr[3]))*2**4 + (int(s21_sr[4]))*2**3 + (int(s21_sr[5]))*2**2 + (int(s21_sr[6]))*2**1 + (int(s21_sr[7]))*2**0
s22_sr_10 = (int(s22_sr[0]))*2**7 + (int(s22_sr[1]))*2**6 + (int(s22_sr[2]))*2**5 + (int(s22_sr[3]))*2**4 + (int(s22_sr[4]))*2**3 + (int(s22_sr[5]))*2**2 + (int(s22_sr[6]))*2**1 + (int(s22_sr[7]))*2**0
s23_sr_10 = (int(s23_sr[0]))*2**7 + (int(s23_sr[1]))*2**6 + (int(s23_sr[2]))*2**5 + (int(s23_sr[3]))*2**4 + (int(s23_sr[4]))*2**3 + (int(s23_sr[5]))*2**2 + (int(s23_sr[6]))*2**1 + (int(s23_sr[7]))*2**0
s30_sr_10 = (int(s30_sr[0]))*2**7 + (int(s30_sr[1]))*2**6 + (int(s30_sr[2]))*2**5 + (int(s30_sr[3]))*2**4 + (int(s30_sr[4]))*2**3 + (int(s30_sr[5]))*2**2 + (int(s30_sr[6]))*2**1 + (int(s30_sr[7]))*2**0
s31_sr_10 = (int(s31_sr[0]))*2**7 + (int(s31_sr[1]))*2**6 + (int(s31_sr[2]))*2**5 + (int(s31_sr[3]))*2**4 + (int(s31_sr[4]))*2**3 + (int(s31_sr[5]))*2**2 + (int(s31_sr[6]))*2**1 + (int(s31_sr[7]))*2**0
s32_sr_10 = (int(s32_sr[0]))*2**7 + (int(s32_sr[1]))*2**6 + (int(s32_sr[2]))*2**5 + (int(s32_sr[3]))*2**4 + (int(s32_sr[4]))*2**3 + (int(s32_sr[5]))*2**2 + (int(s32_sr[6]))*2**1 + (int(s32_sr[7]))*2**0
s33_sr_10 = (int(s33_sr[0]))*2**7 + (int(s33_sr[1]))*2**6 + (int(s33_sr[2]))*2**5 + (int(s33_sr[3]))*2**4 + (int(s33_sr[4]))*2**3 + (int(s33_sr[5]))*2**2 + (int(s33_sr[6]))*2**1 + (int(s33_sr[7]))*2**0
# これを基に、MixColumns後の各要素(例外処理を除く)を計算する
s00 = 2*(s00_sr_10) + 3*(s10_sr_10) + 1*(s20_sr_10) + 1*(s30_sr_10)
s01 = 2*(s01_sr_10) + 3*(s11_sr_10) + 1*(s21_sr_10) + 1*(s31_sr_10)
s02 = 2*(s02_sr_10) + 3*(s12_sr_10) + 1*(s22_sr_10) + 1*(s32_sr_10)
s03 = 2*(s03_sr_10) + 3*(s13_sr_10) + 1*(s23_sr_10) + 1*(s33_sr_10)
s10 = 1*(s00_sr_10) + 2*(s10_sr_10) + 3*(s20_sr_10) + 1*(s30_sr_10)
s11 = 1*(s01_sr_10) + 2*(s11_sr_10) + 3*(s21_sr_10) + 1*(s31_sr_10)
s12 = 1*(s02_sr_10) + 2*(s12_sr_10) + 3*(s22_sr_10) + 1*(s33_sr_10)
s13 = 1*(s03_sr_10) + 2*(s13_sr_10) + 3*(s23_sr_10) + 1*(s33_sr_10)
s20 = 1*(s00_sr_10) + 1*(s10_sr_10) + 2*(s20_sr_10) + 3*(s30_sr_10)
s21 = 1*(s01_sr_10) + 1*(s11_sr_10) + 2*(s21_sr_10) + 3*(s31_sr_10)
s22 = 1*(s02_sr_10) + 1*(s12_sr_10) + 2*(s22_sr_10) + 3*(s32_sr_10)
s23 = 1*(s03_sr_10) + 1*(s13_sr_10) + 2*(s23_sr_10) + 3*(s33_sr_10)
s30 = 3*(s00_sr_10) + 1*(s10_sr_10) + 1*(s20_sr_10) + 2*(s30_sr_10)
s31 = 3*(s01_sr_10) + 1*(s11_sr_10) + 1*(s21_sr_10) + 2*(s31_sr_10)
s32 = 3*(s02_sr_10) + 1*(s12_sr_10) + 1*(s22_sr_10) + 2*(s32_sr_10)
s33 = 3*(s03_sr_10) + 1*(s13_sr_10) + 1*(s23_sr_10) + 2*(s33_sr_10)
# 0bを付けない形式の2進数に直す
s00 = format(s00, 'b')
s01 = format(s01, 'b')
s02 = format(s02, 'b')
s03 = format(s03, 'b')
s10 = format(s10, 'b')
s11 = format(s11, 'b')
s12 = format(s12, 'b')
s13 = format(s13, 'b')
s20 = format(s20, 'b')
s21 = format(s21, 'b')
s22 = format(s22, 'b')
s23 = format(s23, 'b')
s30 = format(s30, 'b')
s31 = format(s31, 'b')
s32 = format(s32, 'b')
s33 = format(s33, 'b')
# 8桁に満たない場合、先頭に0を付けて8桁に揃える
if (len(s00) < 8):
    for i in range(8-len(s00)):
        s00 = '0' + s00
if (len(s01) < 8):
    for i in range(8-len(s01)):
        s01 = '0' + s01     
if (len(s02) < 8):
    for i in range(8-len(s02)):
        s02 = '0' + s02
if (len(s03) < 8):
    for i in range(8-len(s03)):
        s03 = '0' + s03
if (len(s10) < 8):
    for i in range(8-len(s10)):
        s10 = '0' + s10
if (len(s11) < 8):
    for i in range(8-len(s11)):
        s11 = '0' + s11
if (len(s12) < 8):
    for i in range(8-len(s12)):
        s12 = '0' + s12
if (len(s13) < 8):
    for i in range(8-len(s13)):
        s13 = '0' + s13
if (len(s20) < 8):
    for i in range(8-len(s20)):
        s20 = '0' + s20
if (len(s21) < 8):
    for i in range(8-len(s21)):
        s21 = '0' + s21
if (len(s22) < 8):
    for i in range(8-len(s22)):
        s22 = '0' + s22
if (len(s23) < 8):
    for i in range(8-len(s23)):
        s23 = '0' + s23
if (len(s30) < 8):
    for i in range(8-len(s30)):
        s30 = '0' + s30
if (len(s31) < 8):
    for i in range(8-len(s31)):
        s31 = '0' + s31
if (len(s32) < 8):
    for i in range(8-len(s32)):
        s32 = '0' + s32
if (len(s33) < 8):
    for i in range(8-len(s33)):
        s33 = '0' + s33
# 例外処理
##########
# s00
##########
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]

##########
# s01
##########
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]

##########
# s02
##########
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]

##########
# s03
##########
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]

##########
# s10
##########
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]

##########
# s11
##########
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]

##########
# s12
##########
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]

##########
# s13
##########
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]

##########
# s20
##########
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]

##########
# s21
##########
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]

##########
# s22
##########
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]

##########
# s23
##########
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]

##########
# s30
##########
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]

##########
# s31
##########
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]

##########
# s32
##########
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]

##########
# s33
##########
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
# これにてMixColumnsが終了

# AddRoundKey(s, ki)
# k4行列の各要素を取得
emp_list = []
for i in range(0, 8):
    emp_list.append(k4[i])
k00 = ''.join(emp_list)
emp_list = []
for i in range(8, 16):
    emp_list.append(k4[i])
k10 = ''.join(emp_list)
emp_list = []
for i in range(16, 24):
    emp_list.append(k4[i])
k20 = ''.join(emp_list)
emp_list = []
for i in range(24, 32):
    emp_list.append(k4[i])
k30 = ''.join(emp_list)
emp_list = []
for i in range(32, 40):
    emp_list.append(k4[i])
k01 = ''.join(emp_list)
emp_list = []
for i in range(40, 48):
    emp_list.append(k4[i])
k11 = ''.join(emp_list)
emp_list = []
for i in range(48, 56):
    emp_list.append(k4[i])
k21 = ''.join(emp_list)
emp_list = []
for i in range(56, 64):
    emp_list.append(k4[i])
k31 = ''.join(emp_list)
emp_list = []
for i in range(64, 72):
    emp_list.append(k4[i])
k02 = ''.join(emp_list)
emp_list = []
for i in range(72, 80):
    emp_list.append(k4[i])
k12 = ''.join(emp_list)
emp_list = []
for i in range(80, 88):
    emp_list.append(k4[i])
k22 = ''.join(emp_list)
emp_list = []
for i in range(88, 96):
    emp_list.append(k4[i])
k32 = ''.join(emp_list)
emp_list = []
for i in range(96, 104):
    emp_list.append(k4[i])
k03 = ''.join(emp_list)
emp_list = []
for i in range(104, 112):
    emp_list.append(k4[i])
k13 = ''.join(emp_list)
emp_list = []
for i in range(112, 120):
    emp_list.append(k4[i])
k23 = ''.join(emp_list)
emp_list = []
for i in range(120, 128):
    emp_list.append(k4[i])
k33 = ''.join(emp_list)

# 対応する要素ごとに排他的論理和を取る
xor = [ord(a) ^ ord(b) for a,b in zip(s00, k00)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s00 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s10, k10)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s10 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s20, k20)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s20 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s30, k30)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s30 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s01, k01)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s01 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s11, k11)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s11 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s21, k21)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s21 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s31, k31)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s31 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s02, k02)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s02 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s12, k12)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s12 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s22, k22)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s22 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s32, k32)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s32 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s03, k03)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s03 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s13, k13)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s13 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s23, k23)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s23 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s33, k33)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s33 = ''.join(xor_new)


####################
# i = 5
####################
# SubBytes(s)
s00 = sbox[s00]
s10 = sbox[s10]
s20 = sbox[s20]
s30 = sbox[s30]
s01 = sbox[s01]
s11 = sbox[s11]
s21 = sbox[s21]
s31 = sbox[s31]
s02 = sbox[s02]
s12 = sbox[s12]
s22 = sbox[s22]
s32 = sbox[s32]
s03 = sbox[s03]
s13 = sbox[s13]
s23 = sbox[s23]
s33 = sbox[s33]

# ShiftRows(s)
s00_sr = s00
s01_sr = s01
s02_sr = s02
s03_sr = s03
s10_sr = s11
s11_sr = s12
s12_sr = s13
s13_sr = s10
s20_sr = s22
s21_sr = s23
s22_sr = s20
s23_sr = s21
s30_sr = s33
s31_sr = s30
s32_sr = s31
s33_sr = s32

# MixColumns(s)
# 元のsの各要素を10進数に戻す
s00_sr_10 = (int(s00_sr[0]))*2**7 + (int(s00_sr[1]))*2**6 + (int(s00_sr[2]))*2**5 + (int(s00_sr[3]))*2**4 + (int(s00_sr[4]))*2**3 + (int(s00_sr[5]))*2**2 + (int(s00_sr[6]))*2**1 + (int(s00_sr[7]))*2**0
s01_sr_10 = (int(s01_sr[0]))*2**7 + (int(s01_sr[1]))*2**6 + (int(s01_sr[2]))*2**5 + (int(s01_sr[3]))*2**4 + (int(s01_sr[4]))*2**3 + (int(s01_sr[5]))*2**2 + (int(s01_sr[6]))*2**1 + (int(s01_sr[7]))*2**0
s02_sr_10 = (int(s02_sr[0]))*2**7 + (int(s02_sr[1]))*2**6 + (int(s02_sr[2]))*2**5 + (int(s02_sr[3]))*2**4 + (int(s02_sr[4]))*2**3 + (int(s02_sr[5]))*2**2 + (int(s02_sr[6]))*2**1 + (int(s02_sr[7]))*2**0
s03_sr_10 = (int(s03_sr[0]))*2**7 + (int(s03_sr[1]))*2**6 + (int(s03_sr[2]))*2**5 + (int(s03_sr[3]))*2**4 + (int(s03_sr[4]))*2**3 + (int(s03_sr[5]))*2**2 + (int(s03_sr[6]))*2**1 + (int(s03_sr[7]))*2**0
s10_sr_10 = (int(s10_sr[0]))*2**7 + (int(s10_sr[1]))*2**6 + (int(s10_sr[2]))*2**5 + (int(s10_sr[3]))*2**4 + (int(s10_sr[4]))*2**3 + (int(s10_sr[5]))*2**2 + (int(s10_sr[6]))*2**1 + (int(s10_sr[7]))*2**0
s11_sr_10 = (int(s11_sr[0]))*2**7 + (int(s11_sr[1]))*2**6 + (int(s11_sr[2]))*2**5 + (int(s11_sr[3]))*2**4 + (int(s11_sr[4]))*2**3 + (int(s11_sr[5]))*2**2 + (int(s11_sr[6]))*2**1 + (int(s11_sr[7]))*2**0
s12_sr_10 = (int(s12_sr[0]))*2**7 + (int(s12_sr[1]))*2**6 + (int(s12_sr[2]))*2**5 + (int(s12_sr[3]))*2**4 + (int(s12_sr[4]))*2**3 + (int(s12_sr[5]))*2**2 + (int(s12_sr[6]))*2**1 + (int(s12_sr[7]))*2**0
s13_sr_10 = (int(s13_sr[0]))*2**7 + (int(s13_sr[1]))*2**6 + (int(s13_sr[2]))*2**5 + (int(s13_sr[3]))*2**4 + (int(s13_sr[4]))*2**3 + (int(s13_sr[5]))*2**2 + (int(s13_sr[6]))*2**1 + (int(s13_sr[7]))*2**0
s20_sr_10 = (int(s20_sr[0]))*2**7 + (int(s20_sr[1]))*2**6 + (int(s20_sr[2]))*2**5 + (int(s20_sr[3]))*2**4 + (int(s20_sr[4]))*2**3 + (int(s20_sr[5]))*2**2 + (int(s20_sr[6]))*2**1 + (int(s20_sr[7]))*2**0
s21_sr_10 = (int(s21_sr[0]))*2**7 + (int(s21_sr[1]))*2**6 + (int(s21_sr[2]))*2**5 + (int(s21_sr[3]))*2**4 + (int(s21_sr[4]))*2**3 + (int(s21_sr[5]))*2**2 + (int(s21_sr[6]))*2**1 + (int(s21_sr[7]))*2**0
s22_sr_10 = (int(s22_sr[0]))*2**7 + (int(s22_sr[1]))*2**6 + (int(s22_sr[2]))*2**5 + (int(s22_sr[3]))*2**4 + (int(s22_sr[4]))*2**3 + (int(s22_sr[5]))*2**2 + (int(s22_sr[6]))*2**1 + (int(s22_sr[7]))*2**0
s23_sr_10 = (int(s23_sr[0]))*2**7 + (int(s23_sr[1]))*2**6 + (int(s23_sr[2]))*2**5 + (int(s23_sr[3]))*2**4 + (int(s23_sr[4]))*2**3 + (int(s23_sr[5]))*2**2 + (int(s23_sr[6]))*2**1 + (int(s23_sr[7]))*2**0
s30_sr_10 = (int(s30_sr[0]))*2**7 + (int(s30_sr[1]))*2**6 + (int(s30_sr[2]))*2**5 + (int(s30_sr[3]))*2**4 + (int(s30_sr[4]))*2**3 + (int(s30_sr[5]))*2**2 + (int(s30_sr[6]))*2**1 + (int(s30_sr[7]))*2**0
s31_sr_10 = (int(s31_sr[0]))*2**7 + (int(s31_sr[1]))*2**6 + (int(s31_sr[2]))*2**5 + (int(s31_sr[3]))*2**4 + (int(s31_sr[4]))*2**3 + (int(s31_sr[5]))*2**2 + (int(s31_sr[6]))*2**1 + (int(s31_sr[7]))*2**0
s32_sr_10 = (int(s32_sr[0]))*2**7 + (int(s32_sr[1]))*2**6 + (int(s32_sr[2]))*2**5 + (int(s32_sr[3]))*2**4 + (int(s32_sr[4]))*2**3 + (int(s32_sr[5]))*2**2 + (int(s32_sr[6]))*2**1 + (int(s32_sr[7]))*2**0
s33_sr_10 = (int(s33_sr[0]))*2**7 + (int(s33_sr[1]))*2**6 + (int(s33_sr[2]))*2**5 + (int(s33_sr[3]))*2**4 + (int(s33_sr[4]))*2**3 + (int(s33_sr[5]))*2**2 + (int(s33_sr[6]))*2**1 + (int(s33_sr[7]))*2**0
# これを基に、MixColumns後の各要素(例外処理を除く)を計算する
s00 = 2*(s00_sr_10) + 3*(s10_sr_10) + 1*(s20_sr_10) + 1*(s30_sr_10)
s01 = 2*(s01_sr_10) + 3*(s11_sr_10) + 1*(s21_sr_10) + 1*(s31_sr_10)
s02 = 2*(s02_sr_10) + 3*(s12_sr_10) + 1*(s22_sr_10) + 1*(s32_sr_10)
s03 = 2*(s03_sr_10) + 3*(s13_sr_10) + 1*(s23_sr_10) + 1*(s33_sr_10)
s10 = 1*(s00_sr_10) + 2*(s10_sr_10) + 3*(s20_sr_10) + 1*(s30_sr_10)
s11 = 1*(s01_sr_10) + 2*(s11_sr_10) + 3*(s21_sr_10) + 1*(s31_sr_10)
s12 = 1*(s02_sr_10) + 2*(s12_sr_10) + 3*(s22_sr_10) + 1*(s33_sr_10)
s13 = 1*(s03_sr_10) + 2*(s13_sr_10) + 3*(s23_sr_10) + 1*(s33_sr_10)
s20 = 1*(s00_sr_10) + 1*(s10_sr_10) + 2*(s20_sr_10) + 3*(s30_sr_10)
s21 = 1*(s01_sr_10) + 1*(s11_sr_10) + 2*(s21_sr_10) + 3*(s31_sr_10)
s22 = 1*(s02_sr_10) + 1*(s12_sr_10) + 2*(s22_sr_10) + 3*(s32_sr_10)
s23 = 1*(s03_sr_10) + 1*(s13_sr_10) + 2*(s23_sr_10) + 3*(s33_sr_10)
s30 = 3*(s00_sr_10) + 1*(s10_sr_10) + 1*(s20_sr_10) + 2*(s30_sr_10)
s31 = 3*(s01_sr_10) + 1*(s11_sr_10) + 1*(s21_sr_10) + 2*(s31_sr_10)
s32 = 3*(s02_sr_10) + 1*(s12_sr_10) + 1*(s22_sr_10) + 2*(s32_sr_10)
s33 = 3*(s03_sr_10) + 1*(s13_sr_10) + 1*(s23_sr_10) + 2*(s33_sr_10)
# 0bを付けない形式の2進数に直す
s00 = format(s00, 'b')
s01 = format(s01, 'b')
s02 = format(s02, 'b')
s03 = format(s03, 'b')
s10 = format(s10, 'b')
s11 = format(s11, 'b')
s12 = format(s12, 'b')
s13 = format(s13, 'b')
s20 = format(s20, 'b')
s21 = format(s21, 'b')
s22 = format(s22, 'b')
s23 = format(s23, 'b')
s30 = format(s30, 'b')
s31 = format(s31, 'b')
s32 = format(s32, 'b')
s33 = format(s33, 'b')
# 8桁に満たない場合、先頭に0を付けて8桁に揃える
if (len(s00) < 8):
    for i in range(8-len(s00)):
        s00 = '0' + s00
if (len(s01) < 8):
    for i in range(8-len(s01)):
        s01 = '0' + s01     
if (len(s02) < 8):
    for i in range(8-len(s02)):
        s02 = '0' + s02
if (len(s03) < 8):
    for i in range(8-len(s03)):
        s03 = '0' + s03
if (len(s10) < 8):
    for i in range(8-len(s10)):
        s10 = '0' + s10
if (len(s11) < 8):
    for i in range(8-len(s11)):
        s11 = '0' + s11
if (len(s12) < 8):
    for i in range(8-len(s12)):
        s12 = '0' + s12
if (len(s13) < 8):
    for i in range(8-len(s13)):
        s13 = '0' + s13
if (len(s20) < 8):
    for i in range(8-len(s20)):
        s20 = '0' + s20
if (len(s21) < 8):
    for i in range(8-len(s21)):
        s21 = '0' + s21
if (len(s22) < 8):
    for i in range(8-len(s22)):
        s22 = '0' + s22
if (len(s23) < 8):
    for i in range(8-len(s23)):
        s23 = '0' + s23
if (len(s30) < 8):
    for i in range(8-len(s30)):
        s30 = '0' + s30
if (len(s31) < 8):
    for i in range(8-len(s31)):
        s31 = '0' + s31
if (len(s32) < 8):
    for i in range(8-len(s32)):
        s32 = '0' + s32
if (len(s33) < 8):
    for i in range(8-len(s33)):
        s33 = '0' + s33
# 例外処理
##########
# s00
##########
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]

##########
# s01
##########
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]

##########
# s02
##########
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]

##########
# s03
##########
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]

##########
# s10
##########
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]

##########
# s11
##########
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]

##########
# s12
##########
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]

##########
# s13
##########
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]

##########
# s20
##########
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]

##########
# s21
##########
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]

##########
# s22
##########
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]

##########
# s23
##########
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]

##########
# s30
##########
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]

##########
# s31
##########
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]

##########
# s32
##########
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]

##########
# s33
##########
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
# これにてMixColumnsが終了

# AddRoundKey(s, ki)
# k5行列の各要素を取得
emp_list = []
for i in range(0, 8):
    emp_list.append(k5[i])
k00 = ''.join(emp_list)
emp_list = []
for i in range(8, 16):
    emp_list.append(k5[i])
k10 = ''.join(emp_list)
emp_list = []
for i in range(16, 24):
    emp_list.append(k5[i])
k20 = ''.join(emp_list)
emp_list = []
for i in range(24, 32):
    emp_list.append(k5[i])
k30 = ''.join(emp_list)
emp_list = []
for i in range(32, 40):
    emp_list.append(k5[i])
k01 = ''.join(emp_list)
emp_list = []
for i in range(40, 48):
    emp_list.append(k5[i])
k11 = ''.join(emp_list)
emp_list = []
for i in range(48, 56):
    emp_list.append(k5[i])
k21 = ''.join(emp_list)
emp_list = []
for i in range(56, 64):
    emp_list.append(k5[i])
k31 = ''.join(emp_list)
emp_list = []
for i in range(64, 72):
    emp_list.append(k5[i])
k02 = ''.join(emp_list)
emp_list = []
for i in range(72, 80):
    emp_list.append(k5[i])
k12 = ''.join(emp_list)
emp_list = []
for i in range(80, 88):
    emp_list.append(k5[i])
k22 = ''.join(emp_list)
emp_list = []
for i in range(88, 96):
    emp_list.append(k5[i])
k32 = ''.join(emp_list)
emp_list = []
for i in range(96, 104):
    emp_list.append(k5[i])
k03 = ''.join(emp_list)
emp_list = []
for i in range(104, 112):
    emp_list.append(k5[i])
k13 = ''.join(emp_list)
emp_list = []
for i in range(112, 120):
    emp_list.append(k5[i])
k23 = ''.join(emp_list)
emp_list = []
for i in range(120, 128):
    emp_list.append(k5[i])
k33 = ''.join(emp_list)

# 対応する要素ごとに排他的論理和を取る
xor = [ord(a) ^ ord(b) for a,b in zip(s00, k00)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s00 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s10, k10)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s10 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s20, k20)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s20 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s30, k30)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s30 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s01, k01)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s01 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s11, k11)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s11 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s21, k21)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s21 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s31, k31)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s31 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s02, k02)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s02 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s12, k12)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s12 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s22, k22)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s22 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s32, k32)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s32 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s03, k03)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s03 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s13, k13)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s13 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s23, k23)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s23 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s33, k33)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s33 = ''.join(xor_new)


####################
# i = 6
####################
# SubBytes(s)
s00 = sbox[s00]
s10 = sbox[s10]
s20 = sbox[s20]
s30 = sbox[s30]
s01 = sbox[s01]
s11 = sbox[s11]
s21 = sbox[s21]
s31 = sbox[s31]
s02 = sbox[s02]
s12 = sbox[s12]
s22 = sbox[s22]
s32 = sbox[s32]
s03 = sbox[s03]
s13 = sbox[s13]
s23 = sbox[s23]
s33 = sbox[s33]

# ShiftRows(s)
s00_sr = s00
s01_sr = s01
s02_sr = s02
s03_sr = s03
s10_sr = s11
s11_sr = s12
s12_sr = s13
s13_sr = s10
s20_sr = s22
s21_sr = s23
s22_sr = s20
s23_sr = s21
s30_sr = s33
s31_sr = s30
s32_sr = s31
s33_sr = s32

# MixColumns(s)
# 元のsの各要素を10進数に戻す
s00_sr_10 = (int(s00_sr[0]))*2**7 + (int(s00_sr[1]))*2**6 + (int(s00_sr[2]))*2**5 + (int(s00_sr[3]))*2**4 + (int(s00_sr[4]))*2**3 + (int(s00_sr[5]))*2**2 + (int(s00_sr[6]))*2**1 + (int(s00_sr[7]))*2**0
s01_sr_10 = (int(s01_sr[0]))*2**7 + (int(s01_sr[1]))*2**6 + (int(s01_sr[2]))*2**5 + (int(s01_sr[3]))*2**4 + (int(s01_sr[4]))*2**3 + (int(s01_sr[5]))*2**2 + (int(s01_sr[6]))*2**1 + (int(s01_sr[7]))*2**0
s02_sr_10 = (int(s02_sr[0]))*2**7 + (int(s02_sr[1]))*2**6 + (int(s02_sr[2]))*2**5 + (int(s02_sr[3]))*2**4 + (int(s02_sr[4]))*2**3 + (int(s02_sr[5]))*2**2 + (int(s02_sr[6]))*2**1 + (int(s02_sr[7]))*2**0
s03_sr_10 = (int(s03_sr[0]))*2**7 + (int(s03_sr[1]))*2**6 + (int(s03_sr[2]))*2**5 + (int(s03_sr[3]))*2**4 + (int(s03_sr[4]))*2**3 + (int(s03_sr[5]))*2**2 + (int(s03_sr[6]))*2**1 + (int(s03_sr[7]))*2**0
s10_sr_10 = (int(s10_sr[0]))*2**7 + (int(s10_sr[1]))*2**6 + (int(s10_sr[2]))*2**5 + (int(s10_sr[3]))*2**4 + (int(s10_sr[4]))*2**3 + (int(s10_sr[5]))*2**2 + (int(s10_sr[6]))*2**1 + (int(s10_sr[7]))*2**0
s11_sr_10 = (int(s11_sr[0]))*2**7 + (int(s11_sr[1]))*2**6 + (int(s11_sr[2]))*2**5 + (int(s11_sr[3]))*2**4 + (int(s11_sr[4]))*2**3 + (int(s11_sr[5]))*2**2 + (int(s11_sr[6]))*2**1 + (int(s11_sr[7]))*2**0
s12_sr_10 = (int(s12_sr[0]))*2**7 + (int(s12_sr[1]))*2**6 + (int(s12_sr[2]))*2**5 + (int(s12_sr[3]))*2**4 + (int(s12_sr[4]))*2**3 + (int(s12_sr[5]))*2**2 + (int(s12_sr[6]))*2**1 + (int(s12_sr[7]))*2**0
s13_sr_10 = (int(s13_sr[0]))*2**7 + (int(s13_sr[1]))*2**6 + (int(s13_sr[2]))*2**5 + (int(s13_sr[3]))*2**4 + (int(s13_sr[4]))*2**3 + (int(s13_sr[5]))*2**2 + (int(s13_sr[6]))*2**1 + (int(s13_sr[7]))*2**0
s20_sr_10 = (int(s20_sr[0]))*2**7 + (int(s20_sr[1]))*2**6 + (int(s20_sr[2]))*2**5 + (int(s20_sr[3]))*2**4 + (int(s20_sr[4]))*2**3 + (int(s20_sr[5]))*2**2 + (int(s20_sr[6]))*2**1 + (int(s20_sr[7]))*2**0
s21_sr_10 = (int(s21_sr[0]))*2**7 + (int(s21_sr[1]))*2**6 + (int(s21_sr[2]))*2**5 + (int(s21_sr[3]))*2**4 + (int(s21_sr[4]))*2**3 + (int(s21_sr[5]))*2**2 + (int(s21_sr[6]))*2**1 + (int(s21_sr[7]))*2**0
s22_sr_10 = (int(s22_sr[0]))*2**7 + (int(s22_sr[1]))*2**6 + (int(s22_sr[2]))*2**5 + (int(s22_sr[3]))*2**4 + (int(s22_sr[4]))*2**3 + (int(s22_sr[5]))*2**2 + (int(s22_sr[6]))*2**1 + (int(s22_sr[7]))*2**0
s23_sr_10 = (int(s23_sr[0]))*2**7 + (int(s23_sr[1]))*2**6 + (int(s23_sr[2]))*2**5 + (int(s23_sr[3]))*2**4 + (int(s23_sr[4]))*2**3 + (int(s23_sr[5]))*2**2 + (int(s23_sr[6]))*2**1 + (int(s23_sr[7]))*2**0
s30_sr_10 = (int(s30_sr[0]))*2**7 + (int(s30_sr[1]))*2**6 + (int(s30_sr[2]))*2**5 + (int(s30_sr[3]))*2**4 + (int(s30_sr[4]))*2**3 + (int(s30_sr[5]))*2**2 + (int(s30_sr[6]))*2**1 + (int(s30_sr[7]))*2**0
s31_sr_10 = (int(s31_sr[0]))*2**7 + (int(s31_sr[1]))*2**6 + (int(s31_sr[2]))*2**5 + (int(s31_sr[3]))*2**4 + (int(s31_sr[4]))*2**3 + (int(s31_sr[5]))*2**2 + (int(s31_sr[6]))*2**1 + (int(s31_sr[7]))*2**0
s32_sr_10 = (int(s32_sr[0]))*2**7 + (int(s32_sr[1]))*2**6 + (int(s32_sr[2]))*2**5 + (int(s32_sr[3]))*2**4 + (int(s32_sr[4]))*2**3 + (int(s32_sr[5]))*2**2 + (int(s32_sr[6]))*2**1 + (int(s32_sr[7]))*2**0
s33_sr_10 = (int(s33_sr[0]))*2**7 + (int(s33_sr[1]))*2**6 + (int(s33_sr[2]))*2**5 + (int(s33_sr[3]))*2**4 + (int(s33_sr[4]))*2**3 + (int(s33_sr[5]))*2**2 + (int(s33_sr[6]))*2**1 + (int(s33_sr[7]))*2**0
# これを基に、MixColumns後の各要素(例外処理を除く)を計算する
s00 = 2*(s00_sr_10) + 3*(s10_sr_10) + 1*(s20_sr_10) + 1*(s30_sr_10)
s01 = 2*(s01_sr_10) + 3*(s11_sr_10) + 1*(s21_sr_10) + 1*(s31_sr_10)
s02 = 2*(s02_sr_10) + 3*(s12_sr_10) + 1*(s22_sr_10) + 1*(s32_sr_10)
s03 = 2*(s03_sr_10) + 3*(s13_sr_10) + 1*(s23_sr_10) + 1*(s33_sr_10)
s10 = 1*(s00_sr_10) + 2*(s10_sr_10) + 3*(s20_sr_10) + 1*(s30_sr_10)
s11 = 1*(s01_sr_10) + 2*(s11_sr_10) + 3*(s21_sr_10) + 1*(s31_sr_10)
s12 = 1*(s02_sr_10) + 2*(s12_sr_10) + 3*(s22_sr_10) + 1*(s33_sr_10)
s13 = 1*(s03_sr_10) + 2*(s13_sr_10) + 3*(s23_sr_10) + 1*(s33_sr_10)
s20 = 1*(s00_sr_10) + 1*(s10_sr_10) + 2*(s20_sr_10) + 3*(s30_sr_10)
s21 = 1*(s01_sr_10) + 1*(s11_sr_10) + 2*(s21_sr_10) + 3*(s31_sr_10)
s22 = 1*(s02_sr_10) + 1*(s12_sr_10) + 2*(s22_sr_10) + 3*(s32_sr_10)
s23 = 1*(s03_sr_10) + 1*(s13_sr_10) + 2*(s23_sr_10) + 3*(s33_sr_10)
s30 = 3*(s00_sr_10) + 1*(s10_sr_10) + 1*(s20_sr_10) + 2*(s30_sr_10)
s31 = 3*(s01_sr_10) + 1*(s11_sr_10) + 1*(s21_sr_10) + 2*(s31_sr_10)
s32 = 3*(s02_sr_10) + 1*(s12_sr_10) + 1*(s22_sr_10) + 2*(s32_sr_10)
s33 = 3*(s03_sr_10) + 1*(s13_sr_10) + 1*(s23_sr_10) + 2*(s33_sr_10)
# 0bを付けない形式の2進数に直す
s00 = format(s00, 'b')
s01 = format(s01, 'b')
s02 = format(s02, 'b')
s03 = format(s03, 'b')
s10 = format(s10, 'b')
s11 = format(s11, 'b')
s12 = format(s12, 'b')
s13 = format(s13, 'b')
s20 = format(s20, 'b')
s21 = format(s21, 'b')
s22 = format(s22, 'b')
s23 = format(s23, 'b')
s30 = format(s30, 'b')
s31 = format(s31, 'b')
s32 = format(s32, 'b')
s33 = format(s33, 'b')
# 8桁に満たない場合、先頭に0を付けて8桁に揃える
if (len(s00) < 8):
    for i in range(8-len(s00)):
        s00 = '0' + s00
if (len(s01) < 8):
    for i in range(8-len(s01)):
        s01 = '0' + s01     
if (len(s02) < 8):
    for i in range(8-len(s02)):
        s02 = '0' + s02
if (len(s03) < 8):
    for i in range(8-len(s03)):
        s03 = '0' + s03
if (len(s10) < 8):
    for i in range(8-len(s10)):
        s10 = '0' + s10
if (len(s11) < 8):
    for i in range(8-len(s11)):
        s11 = '0' + s11
if (len(s12) < 8):
    for i in range(8-len(s12)):
        s12 = '0' + s12
if (len(s13) < 8):
    for i in range(8-len(s13)):
        s13 = '0' + s13
if (len(s20) < 8):
    for i in range(8-len(s20)):
        s20 = '0' + s20
if (len(s21) < 8):
    for i in range(8-len(s21)):
        s21 = '0' + s21
if (len(s22) < 8):
    for i in range(8-len(s22)):
        s22 = '0' + s22
if (len(s23) < 8):
    for i in range(8-len(s23)):
        s23 = '0' + s23
if (len(s30) < 8):
    for i in range(8-len(s30)):
        s30 = '0' + s30
if (len(s31) < 8):
    for i in range(8-len(s31)):
        s31 = '0' + s31
if (len(s32) < 8):
    for i in range(8-len(s32)):
        s32 = '0' + s32
if (len(s33) < 8):
    for i in range(8-len(s33)):
        s33 = '0' + s33
# 例外処理
##########
# s00
##########
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]

##########
# s01
##########
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]

##########
# s02
##########
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]

##########
# s03
##########
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]

##########
# s10
##########
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]

##########
# s11
##########
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]

##########
# s12
##########
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]

##########
# s13
##########
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]

##########
# s20
##########
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]

##########
# s21
##########
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]

##########
# s22
##########
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]

##########
# s23
##########
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]

##########
# s30
##########
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]

##########
# s31
##########
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]

##########
# s32
##########
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]

##########
# s33
##########
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
# これにてMixColumnsが終了

# AddRoundKey(s, ki)
# k6行列の各要素を取得
emp_list = []
for i in range(0, 8):
    emp_list.append(k6[i])
k00 = ''.join(emp_list)
emp_list = []
for i in range(8, 16):
    emp_list.append(k6[i])
k10 = ''.join(emp_list)
emp_list = []
for i in range(16, 24):
    emp_list.append(k6[i])
k20 = ''.join(emp_list)
emp_list = []
for i in range(24, 32):
    emp_list.append(k6[i])
k30 = ''.join(emp_list)
emp_list = []
for i in range(32, 40):
    emp_list.append(k6[i])
k01 = ''.join(emp_list)
emp_list = []
for i in range(40, 48):
    emp_list.append(k6[i])
k11 = ''.join(emp_list)
emp_list = []
for i in range(48, 56):
    emp_list.append(k6[i])
k21 = ''.join(emp_list)
emp_list = []
for i in range(56, 64):
    emp_list.append(k6[i])
k31 = ''.join(emp_list)
emp_list = []
for i in range(64, 72):
    emp_list.append(k6[i])
k02 = ''.join(emp_list)
emp_list = []
for i in range(72, 80):
    emp_list.append(k6[i])
k12 = ''.join(emp_list)
emp_list = []
for i in range(80, 88):
    emp_list.append(k6[i])
k22 = ''.join(emp_list)
emp_list = []
for i in range(88, 96):
    emp_list.append(k6[i])
k32 = ''.join(emp_list)
emp_list = []
for i in range(96, 104):
    emp_list.append(k6[i])
k03 = ''.join(emp_list)
emp_list = []
for i in range(104, 112):
    emp_list.append(k6[i])
k13 = ''.join(emp_list)
emp_list = []
for i in range(112, 120):
    emp_list.append(k6[i])
k23 = ''.join(emp_list)
emp_list = []
for i in range(120, 128):
    emp_list.append(k6[i])
k33 = ''.join(emp_list)

# 対応する要素ごとに排他的論理和を取る
xor = [ord(a) ^ ord(b) for a,b in zip(s00, k00)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s00 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s10, k10)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s10 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s20, k20)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s20 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s30, k30)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s30 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s01, k01)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s01 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s11, k11)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s11 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s21, k21)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s21 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s31, k31)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s31 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s02, k02)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s02 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s12, k12)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s12 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s22, k22)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s22 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s32, k32)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s32 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s03, k03)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s03 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s13, k13)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s13 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s23, k23)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s23 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s33, k33)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s33 = ''.join(xor_new)


####################
# i = 7
####################
# SubBytes(s)
s00 = sbox[s00]
s10 = sbox[s10]
s20 = sbox[s20]
s30 = sbox[s30]
s01 = sbox[s01]
s11 = sbox[s11]
s21 = sbox[s21]
s31 = sbox[s31]
s02 = sbox[s02]
s12 = sbox[s12]
s22 = sbox[s22]
s32 = sbox[s32]
s03 = sbox[s03]
s13 = sbox[s13]
s23 = sbox[s23]
s33 = sbox[s33]

# ShiftRows(s)
s00_sr = s00
s01_sr = s01
s02_sr = s02
s03_sr = s03
s10_sr = s11
s11_sr = s12
s12_sr = s13
s13_sr = s10
s20_sr = s22
s21_sr = s23
s22_sr = s20
s23_sr = s21
s30_sr = s33
s31_sr = s30
s32_sr = s31
s33_sr = s32

# MixColumns(s)
# 元のsの各要素を10進数に戻す
s00_sr_10 = (int(s00_sr[0]))*2**7 + (int(s00_sr[1]))*2**6 + (int(s00_sr[2]))*2**5 + (int(s00_sr[3]))*2**4 + (int(s00_sr[4]))*2**3 + (int(s00_sr[5]))*2**2 + (int(s00_sr[6]))*2**1 + (int(s00_sr[7]))*2**0
s01_sr_10 = (int(s01_sr[0]))*2**7 + (int(s01_sr[1]))*2**6 + (int(s01_sr[2]))*2**5 + (int(s01_sr[3]))*2**4 + (int(s01_sr[4]))*2**3 + (int(s01_sr[5]))*2**2 + (int(s01_sr[6]))*2**1 + (int(s01_sr[7]))*2**0
s02_sr_10 = (int(s02_sr[0]))*2**7 + (int(s02_sr[1]))*2**6 + (int(s02_sr[2]))*2**5 + (int(s02_sr[3]))*2**4 + (int(s02_sr[4]))*2**3 + (int(s02_sr[5]))*2**2 + (int(s02_sr[6]))*2**1 + (int(s02_sr[7]))*2**0
s03_sr_10 = (int(s03_sr[0]))*2**7 + (int(s03_sr[1]))*2**6 + (int(s03_sr[2]))*2**5 + (int(s03_sr[3]))*2**4 + (int(s03_sr[4]))*2**3 + (int(s03_sr[5]))*2**2 + (int(s03_sr[6]))*2**1 + (int(s03_sr[7]))*2**0
s10_sr_10 = (int(s10_sr[0]))*2**7 + (int(s10_sr[1]))*2**6 + (int(s10_sr[2]))*2**5 + (int(s10_sr[3]))*2**4 + (int(s10_sr[4]))*2**3 + (int(s10_sr[5]))*2**2 + (int(s10_sr[6]))*2**1 + (int(s10_sr[7]))*2**0
s11_sr_10 = (int(s11_sr[0]))*2**7 + (int(s11_sr[1]))*2**6 + (int(s11_sr[2]))*2**5 + (int(s11_sr[3]))*2**4 + (int(s11_sr[4]))*2**3 + (int(s11_sr[5]))*2**2 + (int(s11_sr[6]))*2**1 + (int(s11_sr[7]))*2**0
s12_sr_10 = (int(s12_sr[0]))*2**7 + (int(s12_sr[1]))*2**6 + (int(s12_sr[2]))*2**5 + (int(s12_sr[3]))*2**4 + (int(s12_sr[4]))*2**3 + (int(s12_sr[5]))*2**2 + (int(s12_sr[6]))*2**1 + (int(s12_sr[7]))*2**0
s13_sr_10 = (int(s13_sr[0]))*2**7 + (int(s13_sr[1]))*2**6 + (int(s13_sr[2]))*2**5 + (int(s13_sr[3]))*2**4 + (int(s13_sr[4]))*2**3 + (int(s13_sr[5]))*2**2 + (int(s13_sr[6]))*2**1 + (int(s13_sr[7]))*2**0
s20_sr_10 = (int(s20_sr[0]))*2**7 + (int(s20_sr[1]))*2**6 + (int(s20_sr[2]))*2**5 + (int(s20_sr[3]))*2**4 + (int(s20_sr[4]))*2**3 + (int(s20_sr[5]))*2**2 + (int(s20_sr[6]))*2**1 + (int(s20_sr[7]))*2**0
s21_sr_10 = (int(s21_sr[0]))*2**7 + (int(s21_sr[1]))*2**6 + (int(s21_sr[2]))*2**5 + (int(s21_sr[3]))*2**4 + (int(s21_sr[4]))*2**3 + (int(s21_sr[5]))*2**2 + (int(s21_sr[6]))*2**1 + (int(s21_sr[7]))*2**0
s22_sr_10 = (int(s22_sr[0]))*2**7 + (int(s22_sr[1]))*2**6 + (int(s22_sr[2]))*2**5 + (int(s22_sr[3]))*2**4 + (int(s22_sr[4]))*2**3 + (int(s22_sr[5]))*2**2 + (int(s22_sr[6]))*2**1 + (int(s22_sr[7]))*2**0
s23_sr_10 = (int(s23_sr[0]))*2**7 + (int(s23_sr[1]))*2**6 + (int(s23_sr[2]))*2**5 + (int(s23_sr[3]))*2**4 + (int(s23_sr[4]))*2**3 + (int(s23_sr[5]))*2**2 + (int(s23_sr[6]))*2**1 + (int(s23_sr[7]))*2**0
s30_sr_10 = (int(s30_sr[0]))*2**7 + (int(s30_sr[1]))*2**6 + (int(s30_sr[2]))*2**5 + (int(s30_sr[3]))*2**4 + (int(s30_sr[4]))*2**3 + (int(s30_sr[5]))*2**2 + (int(s30_sr[6]))*2**1 + (int(s30_sr[7]))*2**0
s31_sr_10 = (int(s31_sr[0]))*2**7 + (int(s31_sr[1]))*2**6 + (int(s31_sr[2]))*2**5 + (int(s31_sr[3]))*2**4 + (int(s31_sr[4]))*2**3 + (int(s31_sr[5]))*2**2 + (int(s31_sr[6]))*2**1 + (int(s31_sr[7]))*2**0
s32_sr_10 = (int(s32_sr[0]))*2**7 + (int(s32_sr[1]))*2**6 + (int(s32_sr[2]))*2**5 + (int(s32_sr[3]))*2**4 + (int(s32_sr[4]))*2**3 + (int(s32_sr[5]))*2**2 + (int(s32_sr[6]))*2**1 + (int(s32_sr[7]))*2**0
s33_sr_10 = (int(s33_sr[0]))*2**7 + (int(s33_sr[1]))*2**6 + (int(s33_sr[2]))*2**5 + (int(s33_sr[3]))*2**4 + (int(s33_sr[4]))*2**3 + (int(s33_sr[5]))*2**2 + (int(s33_sr[6]))*2**1 + (int(s33_sr[7]))*2**0
# これを基に、MixColumns後の各要素(例外処理を除く)を計算する
s00 = 2*(s00_sr_10) + 3*(s10_sr_10) + 1*(s20_sr_10) + 1*(s30_sr_10)
s01 = 2*(s01_sr_10) + 3*(s11_sr_10) + 1*(s21_sr_10) + 1*(s31_sr_10)
s02 = 2*(s02_sr_10) + 3*(s12_sr_10) + 1*(s22_sr_10) + 1*(s32_sr_10)
s03 = 2*(s03_sr_10) + 3*(s13_sr_10) + 1*(s23_sr_10) + 1*(s33_sr_10)
s10 = 1*(s00_sr_10) + 2*(s10_sr_10) + 3*(s20_sr_10) + 1*(s30_sr_10)
s11 = 1*(s01_sr_10) + 2*(s11_sr_10) + 3*(s21_sr_10) + 1*(s31_sr_10)
s12 = 1*(s02_sr_10) + 2*(s12_sr_10) + 3*(s22_sr_10) + 1*(s33_sr_10)
s13 = 1*(s03_sr_10) + 2*(s13_sr_10) + 3*(s23_sr_10) + 1*(s33_sr_10)
s20 = 1*(s00_sr_10) + 1*(s10_sr_10) + 2*(s20_sr_10) + 3*(s30_sr_10)
s21 = 1*(s01_sr_10) + 1*(s11_sr_10) + 2*(s21_sr_10) + 3*(s31_sr_10)
s22 = 1*(s02_sr_10) + 1*(s12_sr_10) + 2*(s22_sr_10) + 3*(s32_sr_10)
s23 = 1*(s03_sr_10) + 1*(s13_sr_10) + 2*(s23_sr_10) + 3*(s33_sr_10)
s30 = 3*(s00_sr_10) + 1*(s10_sr_10) + 1*(s20_sr_10) + 2*(s30_sr_10)
s31 = 3*(s01_sr_10) + 1*(s11_sr_10) + 1*(s21_sr_10) + 2*(s31_sr_10)
s32 = 3*(s02_sr_10) + 1*(s12_sr_10) + 1*(s22_sr_10) + 2*(s32_sr_10)
s33 = 3*(s03_sr_10) + 1*(s13_sr_10) + 1*(s23_sr_10) + 2*(s33_sr_10)
# 0bを付けない形式の2進数に直す
s00 = format(s00, 'b')
s01 = format(s01, 'b')
s02 = format(s02, 'b')
s03 = format(s03, 'b')
s10 = format(s10, 'b')
s11 = format(s11, 'b')
s12 = format(s12, 'b')
s13 = format(s13, 'b')
s20 = format(s20, 'b')
s21 = format(s21, 'b')
s22 = format(s22, 'b')
s23 = format(s23, 'b')
s30 = format(s30, 'b')
s31 = format(s31, 'b')
s32 = format(s32, 'b')
s33 = format(s33, 'b')
# 8桁に満たない場合、先頭に0を付けて8桁に揃える
if (len(s00) < 8):
    for i in range(8-len(s00)):
        s00 = '0' + s00
if (len(s01) < 8):
    for i in range(8-len(s01)):
        s01 = '0' + s01     
if (len(s02) < 8):
    for i in range(8-len(s02)):
        s02 = '0' + s02
if (len(s03) < 8):
    for i in range(8-len(s03)):
        s03 = '0' + s03
if (len(s10) < 8):
    for i in range(8-len(s10)):
        s10 = '0' + s10
if (len(s11) < 8):
    for i in range(8-len(s11)):
        s11 = '0' + s11
if (len(s12) < 8):
    for i in range(8-len(s12)):
        s12 = '0' + s12
if (len(s13) < 8):
    for i in range(8-len(s13)):
        s13 = '0' + s13
if (len(s20) < 8):
    for i in range(8-len(s20)):
        s20 = '0' + s20
if (len(s21) < 8):
    for i in range(8-len(s21)):
        s21 = '0' + s21
if (len(s22) < 8):
    for i in range(8-len(s22)):
        s22 = '0' + s22
if (len(s23) < 8):
    for i in range(8-len(s23)):
        s23 = '0' + s23
if (len(s30) < 8):
    for i in range(8-len(s30)):
        s30 = '0' + s30
if (len(s31) < 8):
    for i in range(8-len(s31)):
        s31 = '0' + s31
if (len(s32) < 8):
    for i in range(8-len(s32)):
        s32 = '0' + s32
if (len(s33) < 8):
    for i in range(8-len(s33)):
        s33 = '0' + s33
# 例外処理
##########
# s00
##########
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]

##########
# s01
##########
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]

##########
# s02
##########
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]

##########
# s03
##########
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]

##########
# s10
##########
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]

##########
# s11
##########
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]

##########
# s12
##########
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]

##########
# s13
##########
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]

##########
# s20
##########
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]

##########
# s21
##########
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]

##########
# s22
##########
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]

##########
# s23
##########
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]

##########
# s30
##########
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]

##########
# s31
##########
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]

##########
# s32
##########
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]

##########
# s33
##########
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
# これにてMixColumnsが終了

# AddRoundKey(s, ki)
# k7行列の各要素を取得
emp_list = []
for i in range(0, 8):
    emp_list.append(k7[i])
k00 = ''.join(emp_list)
emp_list = []
for i in range(8, 16):
    emp_list.append(k7[i])
k10 = ''.join(emp_list)
emp_list = []
for i in range(16, 24):
    emp_list.append(k7[i])
k20 = ''.join(emp_list)
emp_list = []
for i in range(24, 32):
    emp_list.append(k7[i])
k30 = ''.join(emp_list)
emp_list = []
for i in range(32, 40):
    emp_list.append(k7[i])
k01 = ''.join(emp_list)
emp_list = []
for i in range(40, 48):
    emp_list.append(k7[i])
k11 = ''.join(emp_list)
emp_list = []
for i in range(48, 56):
    emp_list.append(k7[i])
k21 = ''.join(emp_list)
emp_list = []
for i in range(56, 64):
    emp_list.append(k7[i])
k31 = ''.join(emp_list)
emp_list = []
for i in range(64, 72):
    emp_list.append(k7[i])
k02 = ''.join(emp_list)
emp_list = []
for i in range(72, 80):
    emp_list.append(k7[i])
k12 = ''.join(emp_list)
emp_list = []
for i in range(80, 88):
    emp_list.append(k7[i])
k22 = ''.join(emp_list)
emp_list = []
for i in range(88, 96):
    emp_list.append(k7[i])
k32 = ''.join(emp_list)
emp_list = []
for i in range(96, 104):
    emp_list.append(k7[i])
k03 = ''.join(emp_list)
emp_list = []
for i in range(104, 112):
    emp_list.append(k7[i])
k13 = ''.join(emp_list)
emp_list = []
for i in range(112, 120):
    emp_list.append(k7[i])
k23 = ''.join(emp_list)
emp_list = []
for i in range(120, 128):
    emp_list.append(k7[i])
k33 = ''.join(emp_list)

# 対応する要素ごとに排他的論理和を取る
xor = [ord(a) ^ ord(b) for a,b in zip(s00, k00)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s00 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s10, k10)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s10 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s20, k20)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s20 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s30, k30)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s30 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s01, k01)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s01 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s11, k11)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s11 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s21, k21)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s21 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s31, k31)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s31 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s02, k02)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s02 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s12, k12)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s12 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s22, k22)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s22 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s32, k32)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s32 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s03, k03)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s03 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s13, k13)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s13 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s23, k23)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s23 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s33, k33)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s33 = ''.join(xor_new)


####################
# i = 8
####################
# SubBytes(s)
s00 = sbox[s00]
s10 = sbox[s10]
s20 = sbox[s20]
s30 = sbox[s30]
s01 = sbox[s01]
s11 = sbox[s11]
s21 = sbox[s21]
s31 = sbox[s31]
s02 = sbox[s02]
s12 = sbox[s12]
s22 = sbox[s22]
s32 = sbox[s32]
s03 = sbox[s03]
s13 = sbox[s13]
s23 = sbox[s23]
s33 = sbox[s33]

# ShiftRows(s)
s00_sr = s00
s01_sr = s01
s02_sr = s02
s03_sr = s03
s10_sr = s11
s11_sr = s12
s12_sr = s13
s13_sr = s10
s20_sr = s22
s21_sr = s23
s22_sr = s20
s23_sr = s21
s30_sr = s33
s31_sr = s30
s32_sr = s31
s33_sr = s32

# MixColumns(s)
# 元のsの各要素を10進数に戻す
s00_sr_10 = (int(s00_sr[0]))*2**7 + (int(s00_sr[1]))*2**6 + (int(s00_sr[2]))*2**5 + (int(s00_sr[3]))*2**4 + (int(s00_sr[4]))*2**3 + (int(s00_sr[5]))*2**2 + (int(s00_sr[6]))*2**1 + (int(s00_sr[7]))*2**0
s01_sr_10 = (int(s01_sr[0]))*2**7 + (int(s01_sr[1]))*2**6 + (int(s01_sr[2]))*2**5 + (int(s01_sr[3]))*2**4 + (int(s01_sr[4]))*2**3 + (int(s01_sr[5]))*2**2 + (int(s01_sr[6]))*2**1 + (int(s01_sr[7]))*2**0
s02_sr_10 = (int(s02_sr[0]))*2**7 + (int(s02_sr[1]))*2**6 + (int(s02_sr[2]))*2**5 + (int(s02_sr[3]))*2**4 + (int(s02_sr[4]))*2**3 + (int(s02_sr[5]))*2**2 + (int(s02_sr[6]))*2**1 + (int(s02_sr[7]))*2**0
s03_sr_10 = (int(s03_sr[0]))*2**7 + (int(s03_sr[1]))*2**6 + (int(s03_sr[2]))*2**5 + (int(s03_sr[3]))*2**4 + (int(s03_sr[4]))*2**3 + (int(s03_sr[5]))*2**2 + (int(s03_sr[6]))*2**1 + (int(s03_sr[7]))*2**0
s10_sr_10 = (int(s10_sr[0]))*2**7 + (int(s10_sr[1]))*2**6 + (int(s10_sr[2]))*2**5 + (int(s10_sr[3]))*2**4 + (int(s10_sr[4]))*2**3 + (int(s10_sr[5]))*2**2 + (int(s10_sr[6]))*2**1 + (int(s10_sr[7]))*2**0
s11_sr_10 = (int(s11_sr[0]))*2**7 + (int(s11_sr[1]))*2**6 + (int(s11_sr[2]))*2**5 + (int(s11_sr[3]))*2**4 + (int(s11_sr[4]))*2**3 + (int(s11_sr[5]))*2**2 + (int(s11_sr[6]))*2**1 + (int(s11_sr[7]))*2**0
s12_sr_10 = (int(s12_sr[0]))*2**7 + (int(s12_sr[1]))*2**6 + (int(s12_sr[2]))*2**5 + (int(s12_sr[3]))*2**4 + (int(s12_sr[4]))*2**3 + (int(s12_sr[5]))*2**2 + (int(s12_sr[6]))*2**1 + (int(s12_sr[7]))*2**0
s13_sr_10 = (int(s13_sr[0]))*2**7 + (int(s13_sr[1]))*2**6 + (int(s13_sr[2]))*2**5 + (int(s13_sr[3]))*2**4 + (int(s13_sr[4]))*2**3 + (int(s13_sr[5]))*2**2 + (int(s13_sr[6]))*2**1 + (int(s13_sr[7]))*2**0
s20_sr_10 = (int(s20_sr[0]))*2**7 + (int(s20_sr[1]))*2**6 + (int(s20_sr[2]))*2**5 + (int(s20_sr[3]))*2**4 + (int(s20_sr[4]))*2**3 + (int(s20_sr[5]))*2**2 + (int(s20_sr[6]))*2**1 + (int(s20_sr[7]))*2**0
s21_sr_10 = (int(s21_sr[0]))*2**7 + (int(s21_sr[1]))*2**6 + (int(s21_sr[2]))*2**5 + (int(s21_sr[3]))*2**4 + (int(s21_sr[4]))*2**3 + (int(s21_sr[5]))*2**2 + (int(s21_sr[6]))*2**1 + (int(s21_sr[7]))*2**0
s22_sr_10 = (int(s22_sr[0]))*2**7 + (int(s22_sr[1]))*2**6 + (int(s22_sr[2]))*2**5 + (int(s22_sr[3]))*2**4 + (int(s22_sr[4]))*2**3 + (int(s22_sr[5]))*2**2 + (int(s22_sr[6]))*2**1 + (int(s22_sr[7]))*2**0
s23_sr_10 = (int(s23_sr[0]))*2**7 + (int(s23_sr[1]))*2**6 + (int(s23_sr[2]))*2**5 + (int(s23_sr[3]))*2**4 + (int(s23_sr[4]))*2**3 + (int(s23_sr[5]))*2**2 + (int(s23_sr[6]))*2**1 + (int(s23_sr[7]))*2**0
s30_sr_10 = (int(s30_sr[0]))*2**7 + (int(s30_sr[1]))*2**6 + (int(s30_sr[2]))*2**5 + (int(s30_sr[3]))*2**4 + (int(s30_sr[4]))*2**3 + (int(s30_sr[5]))*2**2 + (int(s30_sr[6]))*2**1 + (int(s30_sr[7]))*2**0
s31_sr_10 = (int(s31_sr[0]))*2**7 + (int(s31_sr[1]))*2**6 + (int(s31_sr[2]))*2**5 + (int(s31_sr[3]))*2**4 + (int(s31_sr[4]))*2**3 + (int(s31_sr[5]))*2**2 + (int(s31_sr[6]))*2**1 + (int(s31_sr[7]))*2**0
s32_sr_10 = (int(s32_sr[0]))*2**7 + (int(s32_sr[1]))*2**6 + (int(s32_sr[2]))*2**5 + (int(s32_sr[3]))*2**4 + (int(s32_sr[4]))*2**3 + (int(s32_sr[5]))*2**2 + (int(s32_sr[6]))*2**1 + (int(s32_sr[7]))*2**0
s33_sr_10 = (int(s33_sr[0]))*2**7 + (int(s33_sr[1]))*2**6 + (int(s33_sr[2]))*2**5 + (int(s33_sr[3]))*2**4 + (int(s33_sr[4]))*2**3 + (int(s33_sr[5]))*2**2 + (int(s33_sr[6]))*2**1 + (int(s33_sr[7]))*2**0
# これを基に、MixColumns後の各要素(例外処理を除く)を計算する
s00 = 2*(s00_sr_10) + 3*(s10_sr_10) + 1*(s20_sr_10) + 1*(s30_sr_10)
s01 = 2*(s01_sr_10) + 3*(s11_sr_10) + 1*(s21_sr_10) + 1*(s31_sr_10)
s02 = 2*(s02_sr_10) + 3*(s12_sr_10) + 1*(s22_sr_10) + 1*(s32_sr_10)
s03 = 2*(s03_sr_10) + 3*(s13_sr_10) + 1*(s23_sr_10) + 1*(s33_sr_10)
s10 = 1*(s00_sr_10) + 2*(s10_sr_10) + 3*(s20_sr_10) + 1*(s30_sr_10)
s11 = 1*(s01_sr_10) + 2*(s11_sr_10) + 3*(s21_sr_10) + 1*(s31_sr_10)
s12 = 1*(s02_sr_10) + 2*(s12_sr_10) + 3*(s22_sr_10) + 1*(s33_sr_10)
s13 = 1*(s03_sr_10) + 2*(s13_sr_10) + 3*(s23_sr_10) + 1*(s33_sr_10)
s20 = 1*(s00_sr_10) + 1*(s10_sr_10) + 2*(s20_sr_10) + 3*(s30_sr_10)
s21 = 1*(s01_sr_10) + 1*(s11_sr_10) + 2*(s21_sr_10) + 3*(s31_sr_10)
s22 = 1*(s02_sr_10) + 1*(s12_sr_10) + 2*(s22_sr_10) + 3*(s32_sr_10)
s23 = 1*(s03_sr_10) + 1*(s13_sr_10) + 2*(s23_sr_10) + 3*(s33_sr_10)
s30 = 3*(s00_sr_10) + 1*(s10_sr_10) + 1*(s20_sr_10) + 2*(s30_sr_10)
s31 = 3*(s01_sr_10) + 1*(s11_sr_10) + 1*(s21_sr_10) + 2*(s31_sr_10)
s32 = 3*(s02_sr_10) + 1*(s12_sr_10) + 1*(s22_sr_10) + 2*(s32_sr_10)
s33 = 3*(s03_sr_10) + 1*(s13_sr_10) + 1*(s23_sr_10) + 2*(s33_sr_10)
# 0bを付けない形式の2進数に直す
s00 = format(s00, 'b')
s01 = format(s01, 'b')
s02 = format(s02, 'b')
s03 = format(s03, 'b')
s10 = format(s10, 'b')
s11 = format(s11, 'b')
s12 = format(s12, 'b')
s13 = format(s13, 'b')
s20 = format(s20, 'b')
s21 = format(s21, 'b')
s22 = format(s22, 'b')
s23 = format(s23, 'b')
s30 = format(s30, 'b')
s31 = format(s31, 'b')
s32 = format(s32, 'b')
s33 = format(s33, 'b')
# 8桁に満たない場合、先頭に0を付けて8桁に揃える
if (len(s00) < 8):
    for i in range(8-len(s00)):
        s00 = '0' + s00
if (len(s01) < 8):
    for i in range(8-len(s01)):
        s01 = '0' + s01     
if (len(s02) < 8):
    for i in range(8-len(s02)):
        s02 = '0' + s02
if (len(s03) < 8):
    for i in range(8-len(s03)):
        s03 = '0' + s03
if (len(s10) < 8):
    for i in range(8-len(s10)):
        s10 = '0' + s10
if (len(s11) < 8):
    for i in range(8-len(s11)):
        s11 = '0' + s11
if (len(s12) < 8):
    for i in range(8-len(s12)):
        s12 = '0' + s12
if (len(s13) < 8):
    for i in range(8-len(s13)):
        s13 = '0' + s13
if (len(s20) < 8):
    for i in range(8-len(s20)):
        s20 = '0' + s20
if (len(s21) < 8):
    for i in range(8-len(s21)):
        s21 = '0' + s21
if (len(s22) < 8):
    for i in range(8-len(s22)):
        s22 = '0' + s22
if (len(s23) < 8):
    for i in range(8-len(s23)):
        s23 = '0' + s23
if (len(s30) < 8):
    for i in range(8-len(s30)):
        s30 = '0' + s30
if (len(s31) < 8):
    for i in range(8-len(s31)):
        s31 = '0' + s31
if (len(s32) < 8):
    for i in range(8-len(s32)):
        s32 = '0' + s32
if (len(s33) < 8):
    for i in range(8-len(s33)):
        s33 = '0' + s33
# 例外処理
##########
# s00
##########
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]

##########
# s01
##########
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]

##########
# s02
##########
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]

##########
# s03
##########
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]

##########
# s10
##########
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]

##########
# s11
##########
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]

##########
# s12
##########
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]

##########
# s13
##########
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]

##########
# s20
##########
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]

##########
# s21
##########
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]

##########
# s22
##########
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]

##########
# s23
##########
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]

##########
# s30
##########
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]

##########
# s31
##########
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]

##########
# s32
##########
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]

##########
# s33
##########
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
# これにてMixColumnsが終了

# AddRoundKey(s, ki)
# k8行列の各要素を取得
emp_list = []
for i in range(0, 8):
    emp_list.append(k8[i])
k00 = ''.join(emp_list)
emp_list = []
for i in range(8, 16):
    emp_list.append(k8[i])
k10 = ''.join(emp_list)
emp_list = []
for i in range(16, 24):
    emp_list.append(k8[i])
k20 = ''.join(emp_list)
emp_list = []
for i in range(24, 32):
    emp_list.append(k8[i])
k30 = ''.join(emp_list)
emp_list = []
for i in range(32, 40):
    emp_list.append(k8[i])
k01 = ''.join(emp_list)
emp_list = []
for i in range(40, 48):
    emp_list.append(k8[i])
k11 = ''.join(emp_list)
emp_list = []
for i in range(48, 56):
    emp_list.append(k8[i])
k21 = ''.join(emp_list)
emp_list = []
for i in range(56, 64):
    emp_list.append(k8[i])
k31 = ''.join(emp_list)
emp_list = []
for i in range(64, 72):
    emp_list.append(k8[i])
k02 = ''.join(emp_list)
emp_list = []
for i in range(72, 80):
    emp_list.append(k8[i])
k12 = ''.join(emp_list)
emp_list = []
for i in range(80, 88):
    emp_list.append(k8[i])
k22 = ''.join(emp_list)
emp_list = []
for i in range(88, 96):
    emp_list.append(k8[i])
k32 = ''.join(emp_list)
emp_list = []
for i in range(96, 104):
    emp_list.append(k8[i])
k03 = ''.join(emp_list)
emp_list = []
for i in range(104, 112):
    emp_list.append(k8[i])
k13 = ''.join(emp_list)
emp_list = []
for i in range(112, 120):
    emp_list.append(k8[i])
k23 = ''.join(emp_list)
emp_list = []
for i in range(120, 128):
    emp_list.append(k8[i])
k33 = ''.join(emp_list)

# 対応する要素ごとに排他的論理和を取る
xor = [ord(a) ^ ord(b) for a,b in zip(s00, k00)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s00 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s10, k10)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s10 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s20, k20)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s20 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s30, k30)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s30 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s01, k01)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s01 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s11, k11)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s11 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s21, k21)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s21 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s31, k31)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s31 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s02, k02)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s02 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s12, k12)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s12 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s22, k22)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s22 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s32, k32)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s32 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s03, k03)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s03 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s13, k13)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s13 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s23, k23)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s23 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s33, k33)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s33 = ''.join(xor_new)


####################
# i = 9
####################
# SubBytes(s)
s00 = sbox[s00]
s10 = sbox[s10]
s20 = sbox[s20]
s30 = sbox[s30]
s01 = sbox[s01]
s11 = sbox[s11]
s21 = sbox[s21]
s31 = sbox[s31]
s02 = sbox[s02]
s12 = sbox[s12]
s22 = sbox[s22]
s32 = sbox[s32]
s03 = sbox[s03]
s13 = sbox[s13]
s23 = sbox[s23]
s33 = sbox[s33]

# ShiftRows(s)
s00_sr = s00
s01_sr = s01
s02_sr = s02
s03_sr = s03
s10_sr = s11
s11_sr = s12
s12_sr = s13
s13_sr = s10
s20_sr = s22
s21_sr = s23
s22_sr = s20
s23_sr = s21
s30_sr = s33
s31_sr = s30
s32_sr = s31
s33_sr = s32

# MixColumns(s)
# 元のsの各要素を10進数に戻す
s00_sr_10 = (int(s00_sr[0]))*2**7 + (int(s00_sr[1]))*2**6 + (int(s00_sr[2]))*2**5 + (int(s00_sr[3]))*2**4 + (int(s00_sr[4]))*2**3 + (int(s00_sr[5]))*2**2 + (int(s00_sr[6]))*2**1 + (int(s00_sr[7]))*2**0
s01_sr_10 = (int(s01_sr[0]))*2**7 + (int(s01_sr[1]))*2**6 + (int(s01_sr[2]))*2**5 + (int(s01_sr[3]))*2**4 + (int(s01_sr[4]))*2**3 + (int(s01_sr[5]))*2**2 + (int(s01_sr[6]))*2**1 + (int(s01_sr[7]))*2**0
s02_sr_10 = (int(s02_sr[0]))*2**7 + (int(s02_sr[1]))*2**6 + (int(s02_sr[2]))*2**5 + (int(s02_sr[3]))*2**4 + (int(s02_sr[4]))*2**3 + (int(s02_sr[5]))*2**2 + (int(s02_sr[6]))*2**1 + (int(s02_sr[7]))*2**0
s03_sr_10 = (int(s03_sr[0]))*2**7 + (int(s03_sr[1]))*2**6 + (int(s03_sr[2]))*2**5 + (int(s03_sr[3]))*2**4 + (int(s03_sr[4]))*2**3 + (int(s03_sr[5]))*2**2 + (int(s03_sr[6]))*2**1 + (int(s03_sr[7]))*2**0
s10_sr_10 = (int(s10_sr[0]))*2**7 + (int(s10_sr[1]))*2**6 + (int(s10_sr[2]))*2**5 + (int(s10_sr[3]))*2**4 + (int(s10_sr[4]))*2**3 + (int(s10_sr[5]))*2**2 + (int(s10_sr[6]))*2**1 + (int(s10_sr[7]))*2**0
s11_sr_10 = (int(s11_sr[0]))*2**7 + (int(s11_sr[1]))*2**6 + (int(s11_sr[2]))*2**5 + (int(s11_sr[3]))*2**4 + (int(s11_sr[4]))*2**3 + (int(s11_sr[5]))*2**2 + (int(s11_sr[6]))*2**1 + (int(s11_sr[7]))*2**0
s12_sr_10 = (int(s12_sr[0]))*2**7 + (int(s12_sr[1]))*2**6 + (int(s12_sr[2]))*2**5 + (int(s12_sr[3]))*2**4 + (int(s12_sr[4]))*2**3 + (int(s12_sr[5]))*2**2 + (int(s12_sr[6]))*2**1 + (int(s12_sr[7]))*2**0
s13_sr_10 = (int(s13_sr[0]))*2**7 + (int(s13_sr[1]))*2**6 + (int(s13_sr[2]))*2**5 + (int(s13_sr[3]))*2**4 + (int(s13_sr[4]))*2**3 + (int(s13_sr[5]))*2**2 + (int(s13_sr[6]))*2**1 + (int(s13_sr[7]))*2**0
s20_sr_10 = (int(s20_sr[0]))*2**7 + (int(s20_sr[1]))*2**6 + (int(s20_sr[2]))*2**5 + (int(s20_sr[3]))*2**4 + (int(s20_sr[4]))*2**3 + (int(s20_sr[5]))*2**2 + (int(s20_sr[6]))*2**1 + (int(s20_sr[7]))*2**0
s21_sr_10 = (int(s21_sr[0]))*2**7 + (int(s21_sr[1]))*2**6 + (int(s21_sr[2]))*2**5 + (int(s21_sr[3]))*2**4 + (int(s21_sr[4]))*2**3 + (int(s21_sr[5]))*2**2 + (int(s21_sr[6]))*2**1 + (int(s21_sr[7]))*2**0
s22_sr_10 = (int(s22_sr[0]))*2**7 + (int(s22_sr[1]))*2**6 + (int(s22_sr[2]))*2**5 + (int(s22_sr[3]))*2**4 + (int(s22_sr[4]))*2**3 + (int(s22_sr[5]))*2**2 + (int(s22_sr[6]))*2**1 + (int(s22_sr[7]))*2**0
s23_sr_10 = (int(s23_sr[0]))*2**7 + (int(s23_sr[1]))*2**6 + (int(s23_sr[2]))*2**5 + (int(s23_sr[3]))*2**4 + (int(s23_sr[4]))*2**3 + (int(s23_sr[5]))*2**2 + (int(s23_sr[6]))*2**1 + (int(s23_sr[7]))*2**0
s30_sr_10 = (int(s30_sr[0]))*2**7 + (int(s30_sr[1]))*2**6 + (int(s30_sr[2]))*2**5 + (int(s30_sr[3]))*2**4 + (int(s30_sr[4]))*2**3 + (int(s30_sr[5]))*2**2 + (int(s30_sr[6]))*2**1 + (int(s30_sr[7]))*2**0
s31_sr_10 = (int(s31_sr[0]))*2**7 + (int(s31_sr[1]))*2**6 + (int(s31_sr[2]))*2**5 + (int(s31_sr[3]))*2**4 + (int(s31_sr[4]))*2**3 + (int(s31_sr[5]))*2**2 + (int(s31_sr[6]))*2**1 + (int(s31_sr[7]))*2**0
s32_sr_10 = (int(s32_sr[0]))*2**7 + (int(s32_sr[1]))*2**6 + (int(s32_sr[2]))*2**5 + (int(s32_sr[3]))*2**4 + (int(s32_sr[4]))*2**3 + (int(s32_sr[5]))*2**2 + (int(s32_sr[6]))*2**1 + (int(s32_sr[7]))*2**0
s33_sr_10 = (int(s33_sr[0]))*2**7 + (int(s33_sr[1]))*2**6 + (int(s33_sr[2]))*2**5 + (int(s33_sr[3]))*2**4 + (int(s33_sr[4]))*2**3 + (int(s33_sr[5]))*2**2 + (int(s33_sr[6]))*2**1 + (int(s33_sr[7]))*2**0
# これを基に、MixColumns後の各要素(例外処理を除く)を計算する
s00 = 2*(s00_sr_10) + 3*(s10_sr_10) + 1*(s20_sr_10) + 1*(s30_sr_10)
s01 = 2*(s01_sr_10) + 3*(s11_sr_10) + 1*(s21_sr_10) + 1*(s31_sr_10)
s02 = 2*(s02_sr_10) + 3*(s12_sr_10) + 1*(s22_sr_10) + 1*(s32_sr_10)
s03 = 2*(s03_sr_10) + 3*(s13_sr_10) + 1*(s23_sr_10) + 1*(s33_sr_10)
s10 = 1*(s00_sr_10) + 2*(s10_sr_10) + 3*(s20_sr_10) + 1*(s30_sr_10)
s11 = 1*(s01_sr_10) + 2*(s11_sr_10) + 3*(s21_sr_10) + 1*(s31_sr_10)
s12 = 1*(s02_sr_10) + 2*(s12_sr_10) + 3*(s22_sr_10) + 1*(s33_sr_10)
s13 = 1*(s03_sr_10) + 2*(s13_sr_10) + 3*(s23_sr_10) + 1*(s33_sr_10)
s20 = 1*(s00_sr_10) + 1*(s10_sr_10) + 2*(s20_sr_10) + 3*(s30_sr_10)
s21 = 1*(s01_sr_10) + 1*(s11_sr_10) + 2*(s21_sr_10) + 3*(s31_sr_10)
s22 = 1*(s02_sr_10) + 1*(s12_sr_10) + 2*(s22_sr_10) + 3*(s32_sr_10)
s23 = 1*(s03_sr_10) + 1*(s13_sr_10) + 2*(s23_sr_10) + 3*(s33_sr_10)
s30 = 3*(s00_sr_10) + 1*(s10_sr_10) + 1*(s20_sr_10) + 2*(s30_sr_10)
s31 = 3*(s01_sr_10) + 1*(s11_sr_10) + 1*(s21_sr_10) + 2*(s31_sr_10)
s32 = 3*(s02_sr_10) + 1*(s12_sr_10) + 1*(s22_sr_10) + 2*(s32_sr_10)
s33 = 3*(s03_sr_10) + 1*(s13_sr_10) + 1*(s23_sr_10) + 2*(s33_sr_10)
# 0bを付けない形式の2進数に直す
s00 = format(s00, 'b')
s01 = format(s01, 'b')
s02 = format(s02, 'b')
s03 = format(s03, 'b')
s10 = format(s10, 'b')
s11 = format(s11, 'b')
s12 = format(s12, 'b')
s13 = format(s13, 'b')
s20 = format(s20, 'b')
s21 = format(s21, 'b')
s22 = format(s22, 'b')
s23 = format(s23, 'b')
s30 = format(s30, 'b')
s31 = format(s31, 'b')
s32 = format(s32, 'b')
s33 = format(s33, 'b')
# 8桁に満たない場合、先頭に0を付けて8桁に揃える
if (len(s00) < 8):
    for i in range(8-len(s00)):
        s00 = '0' + s00
if (len(s01) < 8):
    for i in range(8-len(s01)):
        s01 = '0' + s01     
if (len(s02) < 8):
    for i in range(8-len(s02)):
        s02 = '0' + s02
if (len(s03) < 8):
    for i in range(8-len(s03)):
        s03 = '0' + s03
if (len(s10) < 8):
    for i in range(8-len(s10)):
        s10 = '0' + s10
if (len(s11) < 8):
    for i in range(8-len(s11)):
        s11 = '0' + s11
if (len(s12) < 8):
    for i in range(8-len(s12)):
        s12 = '0' + s12
if (len(s13) < 8):
    for i in range(8-len(s13)):
        s13 = '0' + s13
if (len(s20) < 8):
    for i in range(8-len(s20)):
        s20 = '0' + s20
if (len(s21) < 8):
    for i in range(8-len(s21)):
        s21 = '0' + s21
if (len(s22) < 8):
    for i in range(8-len(s22)):
        s22 = '0' + s22
if (len(s23) < 8):
    for i in range(8-len(s23)):
        s23 = '0' + s23
if (len(s30) < 8):
    for i in range(8-len(s30)):
        s30 = '0' + s30
if (len(s31) < 8):
    for i in range(8-len(s31)):
        s31 = '0' + s31
if (len(s32) < 8):
    for i in range(8-len(s32)):
        s32 = '0' + s32
if (len(s33) < 8):
    for i in range(8-len(s33)):
        s33 = '0' + s33
# 例外処理
##########
# s00
##########
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]
m_x = '100011011'
if (len(s00) >= len(m_x)):
    for i in range(len(s00) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s00, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s00 = ''.join(xor_new)
    s00 = s00[1:]

##########
# s01
##########
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]
m_x = '100011011'
if (len(s01) >= len(m_x)):
    for i in range(len(s01) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s01, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s01 = ''.join(xor_new)
    s01 = s01[1:]

##########
# s02
##########
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]
m_x = '100011011'
if (len(s02) >= len(m_x)):
    for i in range(len(s02) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s02, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s02 = ''.join(xor_new)
    s02 = s02[1:]

##########
# s03
##########
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]
m_x = '100011011'
if (len(s03) >= len(m_x)):
    for i in range(len(s03) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s03, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s03 = ''.join(xor_new)
    s03 = s03[1:]

##########
# s10
##########
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]
m_x = '100011011'
if (len(s10) >= len(m_x)):
    for i in range(len(s10) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s10, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s10 = ''.join(xor_new)
    s10 = s10[1:]

##########
# s11
##########
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]
m_x = '100011011'
if (len(s11) >= len(m_x)):
    for i in range(len(s11) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s11, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s11 = ''.join(xor_new)
    s11 = s11[1:]

##########
# s12
##########
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]
m_x = '100011011'
if (len(s12) >= len(m_x)):
    for i in range(len(s12) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s12, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s12 = ''.join(xor_new)
    s12 = s12[1:]

##########
# s13
##########
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]
m_x = '100011011'
if (len(s13) >= len(m_x)):
    for i in range(len(s13) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s13, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s13 = ''.join(xor_new)
    s13 = s13[1:]

##########
# s20
##########
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]
m_x = '100011011'
if (len(s20) >= len(m_x)):
    for i in range(len(s20) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s20, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s20 = ''.join(xor_new)
    s20 = s20[1:]

##########
# s21
##########
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]
m_x = '100011011'
if (len(s21) >= len(m_x)):
    for i in range(len(s21) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s21, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s21 = ''.join(xor_new)
    s21 = s21[1:]

##########
# s22
##########
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]
m_x = '100011011'
if (len(s22) >= len(m_x)):
    for i in range(len(s22) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s22, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s22 = ''.join(xor_new)
    s22 = s22[1:]

##########
# s23
##########
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]
m_x = '100011011'
if (len(s23) >= len(m_x)):
    for i in range(len(s23) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s23, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s23 = ''.join(xor_new)
    s23 = s23[1:]

##########
# s30
##########
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]
m_x = '100011011'
if (len(s30) >= len(m_x)):
    for i in range(len(s30) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s30, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s30 = ''.join(xor_new)
    s30 = s30[1:]

##########
# s31
##########
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]
m_x = '100011011'
if (len(s31) >= len(m_x)):
    for i in range(len(s31) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s31, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s31 = ''.join(xor_new)
    s31 = s31[1:]

##########
# s32
##########
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]
m_x = '100011011'
if (len(s32) >= len(m_x)):
    for i in range(len(s32) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s32, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s32 = ''.join(xor_new)
    s32 = s32[1:]

##########
# s33
##########
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
m_x = '100011011'
if (len(s33) >= len(m_x)):
    for i in range(len(s33) - len(m_x)):
        m_x = m_x + '0'
    xor = [ord(a) ^ ord(b) for a,b in zip(s33, m_x)]
    xor_new = []
    for i in range(len(m_x)):
        xor_new.append(str(xor[i]))
    s33 = ''.join(xor_new)
    s33 = s33[1:]
# これにてMixColumnsが終了

# AddRoundKey(s, ki)
# k9行列の各要素を取得
emp_list = []
for i in range(0, 8):
    emp_list.append(k9[i])
k00 = ''.join(emp_list)
emp_list = []
for i in range(8, 16):
    emp_list.append(k9[i])
k10 = ''.join(emp_list)
emp_list = []
for i in range(16, 24):
    emp_list.append(k9[i])
k20 = ''.join(emp_list)
emp_list = []
for i in range(24, 32):
    emp_list.append(k9[i])
k30 = ''.join(emp_list)
emp_list = []
for i in range(32, 40):
    emp_list.append(k9[i])
k01 = ''.join(emp_list)
emp_list = []
for i in range(40, 48):
    emp_list.append(k9[i])
k11 = ''.join(emp_list)
emp_list = []
for i in range(48, 56):
    emp_list.append(k9[i])
k21 = ''.join(emp_list)
emp_list = []
for i in range(56, 64):
    emp_list.append(k9[i])
k31 = ''.join(emp_list)
emp_list = []
for i in range(64, 72):
    emp_list.append(k9[i])
k02 = ''.join(emp_list)
emp_list = []
for i in range(72, 80):
    emp_list.append(k9[i])
k12 = ''.join(emp_list)
emp_list = []
for i in range(80, 88):
    emp_list.append(k9[i])
k22 = ''.join(emp_list)
emp_list = []
for i in range(88, 96):
    emp_list.append(k9[i])
k32 = ''.join(emp_list)
emp_list = []
for i in range(96, 104):
    emp_list.append(k9[i])
k03 = ''.join(emp_list)
emp_list = []
for i in range(104, 112):
    emp_list.append(k9[i])
k13 = ''.join(emp_list)
emp_list = []
for i in range(112, 120):
    emp_list.append(k9[i])
k23 = ''.join(emp_list)
emp_list = []
for i in range(120, 128):
    emp_list.append(k9[i])
k33 = ''.join(emp_list)

# 対応する要素ごとに排他的論理和を取る
xor = [ord(a) ^ ord(b) for a,b in zip(s00, k00)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s00 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s10, k10)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s10 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s20, k20)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s20 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s30, k30)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s30 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s01, k01)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s01 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s11, k11)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s11 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s21, k21)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s21 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s31, k31)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s31 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s02, k02)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s02 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s12, k12)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s12 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s22, k22)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s22 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s32, k32)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s32 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s03, k03)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s03 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s13, k13)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s13 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s23, k23)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s23 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s33, k33)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s33 = ''.join(xor_new)


####################
# i = 10 (最終ラウンド)
####################
# SubBytes(s)
s00 = sbox[s00]
s10 = sbox[s10]
s20 = sbox[s20]
s30 = sbox[s30]
s01 = sbox[s01]
s11 = sbox[s11]
s21 = sbox[s21]
s31 = sbox[s31]
s02 = sbox[s02]
s12 = sbox[s12]
s22 = sbox[s22]
s32 = sbox[s32]
s03 = sbox[s03]
s13 = sbox[s13]
s23 = sbox[s23]
s33 = sbox[s33]

# ShiftRows(s)
s00_sr = s00
s01_sr = s01
s02_sr = s02
s03_sr = s03
s10_sr = s11
s11_sr = s12
s12_sr = s13
s13_sr = s10
s20_sr = s22
s21_sr = s23
s22_sr = s20
s23_sr = s21
s30_sr = s33
s31_sr = s30
s32_sr = s31
s33_sr = s32

# AddRoundKey(s, ki)
# k10行列の各要素を取得
emp_list = []
for i in range(0, 8):
    emp_list.append(k10_subkey[i])
k00 = ''.join(emp_list)
emp_list = []
for i in range(8, 16):
    emp_list.append(k10_subkey[i])
k10 = ''.join(emp_list)
emp_list = []
for i in range(16, 24):
    emp_list.append(k10_subkey[i])
k20 = ''.join(emp_list)
emp_list = []
for i in range(24, 32):
    emp_list.append(k10_subkey[i])
k30 = ''.join(emp_list)
emp_list = []
for i in range(32, 40):
    emp_list.append(k10_subkey[i])
k01 = ''.join(emp_list)
emp_list = []
for i in range(40, 48):
    emp_list.append(k10_subkey[i])
k11 = ''.join(emp_list)
emp_list = []
for i in range(48, 56):
    emp_list.append(k10_subkey[i])
k21 = ''.join(emp_list)
emp_list = []
for i in range(56, 64):
    emp_list.append(k10_subkey[i])
k31 = ''.join(emp_list)
emp_list = []
for i in range(64, 72):
    emp_list.append(k10_subkey[i])
k02 = ''.join(emp_list)
emp_list = []
for i in range(72, 80):
    emp_list.append(k10_subkey[i])
k12 = ''.join(emp_list)
emp_list = []
for i in range(80, 88):
    emp_list.append(k10_subkey[i])
k22 = ''.join(emp_list)
emp_list = []
for i in range(88, 96):
    emp_list.append(k10_subkey[i])
k32 = ''.join(emp_list)
emp_list = []
for i in range(96, 104):
    emp_list.append(k10_subkey[i])
k03 = ''.join(emp_list)
emp_list = []
for i in range(104, 112):
    emp_list.append(k10_subkey[i])
k13 = ''.join(emp_list)
emp_list = []
for i in range(112, 120):
    emp_list.append(k10_subkey[i])
k23 = ''.join(emp_list)
emp_list = []
for i in range(120, 128):
    emp_list.append(k10_subkey[i])
k33 = ''.join(emp_list)

# 対応する要素ごとに排他的論理和を取る
xor = [ord(a) ^ ord(b) for a,b in zip(s00, k00)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s00 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s10, k10)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s10 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s20, k20)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s20 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s30, k30)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s30 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s01, k01)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s01 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s11, k11)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s11 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s21, k21)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s21 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s31, k31)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s31 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s02, k02)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s02 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s12, k12)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s12 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s22, k22)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s22 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s32, k32)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s32 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s03, k03)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s03 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s13, k13)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s13 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s23, k23)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s23 = ''.join(xor_new)
xor = [ord(a) ^ ord(b) for a,b in zip(s33, k33)]
xor_new = []
for i in range(8):
    xor_new.append(str(xor[i]))
s33 = ''.join(xor_new)
####################################################
#################### 暗号化終了 ####################
####################################################


print('暗号化が終了しました。')
print('暗号文は以下の通りです。')

c = s00 + s10 + s20 + s30 + s01 + s11 + s21 + s31 + s02 + s12 + s22 + s32 + s03 + s13 + s23 + s33

print(c)
print('')

# ここまで動作確認済み
# 次やること : 復号




