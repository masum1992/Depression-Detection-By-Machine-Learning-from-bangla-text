filter1_data = open("P:/Study/Thesis works/Masum405/testfile/Depressed words.txt","r", encoding = 'utf8').read()

documents = []

punctuations = '''.,""{}!@$%&=+|`~[]D১২৩৪৫৬৭৮৯০'''


#punctuations = ''','''

for p in filter1_data.split('\n'):
     #print('p = ',str(p))
     f = ""
     all_words = []
     for ch in p:
          if ch not in punctuations:
               f = f + ch
     # print('f = ' + str(f))
     all_words.append(f)
     #print(all_words)
     print(f)
     file_name = "P:/Study/Thesis works/Masum405/testfile/Depressed words22.txt"
     cf = open(file_name, "a", encoding='utf-8')
     cf.write(str(f)+'\n')
cf.close()

