# AES-128
# 平文の上限を128bitに制限したver
# それ以降はもう一度実行

import sys              # 実行環境に関するライブラリ
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

# 動作確認用
print('平文の、ASCIIコードによる2進数化')
print(binmoji)
print('')


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
# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(120, 128):     # シフトする要素を埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 120):      # シフトされた結果、後ろに動く要素を埋める
    binmoji_shift_list.append(binmoji[i])

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

# 排他的論理和(i=1)
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_1, after_okikae_moji)]
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
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3, temp)]
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
# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(120, 128):     # シフトする要素を埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 120):      # シフトされた結果、後ろに動く要素を埋める
    binmoji_shift_list.append(binmoji[i])

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
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_1, after_okikae_moji)]
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
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3, temp)]
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
# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(120, 128):     # シフトする要素を埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 120):      # シフトされた結果、後ろに動く要素を埋める
    binmoji_shift_list.append(binmoji[i])

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
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_1, after_okikae_moji)]
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
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3, temp)]
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
# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(120, 128):     # シフトする要素を埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 120):      # シフトされた結果、後ろに動く要素を埋める
    binmoji_shift_list.append(binmoji[i])

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
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_1, after_okikae_moji)]
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
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3, temp)]
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
# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(120, 128):     # シフトする要素を埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 120):      # シフトされた結果、後ろに動く要素を埋める
    binmoji_shift_list.append(binmoji[i])

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
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_1, after_okikae_moji)]
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
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3, temp)]
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
# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(120, 128):     # シフトする要素を埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 120):      # シフトされた結果、後ろに動く要素を埋める
    binmoji_shift_list.append(binmoji[i])

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
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_1, after_okikae_moji)]
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
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3, temp)]
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
# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(120, 128):     # シフトする要素を埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 120):      # シフトされた結果、後ろに動く要素を埋める
    binmoji_shift_list.append(binmoji[i])

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
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_1, after_okikae_moji)]
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
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3, temp)]
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
# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(120, 128):     # シフトする要素を埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 120):      # シフトされた結果、後ろに動く要素を埋める
    binmoji_shift_list.append(binmoji[i])

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
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_1, after_okikae_moji)]
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
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3, temp)]
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
# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(120, 128):     # シフトする要素を埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 120):      # シフトされた結果、後ろに動く要素を埋める
    binmoji_shift_list.append(binmoji[i])

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
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_1, after_okikae_moji)]
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
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3, temp)]
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
# シフトの実現
binmoji_shift_list = []
for i in range(0, 96):        # 動かさない要素を先に埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(120, 128):     # シフトする要素を埋める
    binmoji_shift_list.append(binmoji[i])
for i in range(96, 120):      # シフトされた結果、後ろに動く要素を埋める
    binmoji_shift_list.append(binmoji[i])

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
xor_1 = [ord(a) ^ ord(b) for a,b in zip(rcon_1, after_okikae_moji)]
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
w7_list = [ord(a) ^ ord(b) for a,b in zip(w3, temp)]
w7_list_str = []
for i in range(32):
    w7_list_str.append(str(w7_list[i]))
w7 = ''.join(w7_list_str)
temp = w7

k10 = w4 + w5 + w6 + w7
binmoji = k10     # binmojiの更新

print('10個目のサブ鍵 k10')
print(k10)

print('サブ鍵生成を終了します。')
print('')



###################################
# データ暗号化アルゴリズム
###################################