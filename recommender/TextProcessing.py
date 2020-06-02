from bert_dp.tokenization import FullTokenizer
import re
import os.path
from pathlib import Path


class TextProcessing:
    def __init__(self):
        home = str(Path.home())
        path = os.path.join(home, ".deeppavlov/downloads/bert_models/multi_cased_L-12_H-768_A-12/vocab.txt")
        self.fTokenizer = FullTokenizer(vocab_file=path,
                                        do_lower_case=False)

    def chunker(self, text):
        """500-s tsuun token baihaar uguulberuuded salgah

        Parameters
        ----------
        text: str
            Jiriin uguulber, text
        Returns
        -------
        list[str]
            500-s baga tokentoi stringuudees togtoh list
        """
        sentences = self.sentence_separator(text)
        tokenLen, sentences = self.tokenize(sentences)
        lenSum = 0
        chunked = []
        chunk = ''
        for i in range(len(sentences)):
            if lenSum + tokenLen[i] < 500:
                lenSum += tokenLen[i]
                chunk += sentences[i]
            else:
                lenSum = tokenLen[i]
                chunked.append(chunk)
                chunk = ''
        if chunk != '':
            chunked.append(chunk)
        return self.clear_empty_elements(chunked)

    def sentence_separator(self, text):
        """Textiig uguulber uguulbereer salgah

        Parameters
        ----------
        text: str

        Returns
        -------
        list[str]
        """
        text = re.sub(' +', ' ', text)  # remove multiple spaces 'The   quick ' -> 'The quick '
        i = 0
        prev_i = 0
        sentence_separated = []
        while len(text) > prev_i:
            space_count = 0
            while text[i:i + 2] not in ['. ', '! ', '? ', '। ', '። ', '။ ', '。 '] and i < len(text):
                # japan / cn text sentence end characters
                if text[i:i + 1] in ['。', '！']:
                    break
                if text[i] == " ":
                    space_count += 1

                if space_count > 60:
                    if text[i:i + 2] in [', ', ': ', '; ', '؟ ', '، ']:
                        break
                    if text[i:i + 1] in ['，', '、', '：']:
                        break
                if space_count > 90:
                    if text[i] == ' ':
                        break
                if space_count < 10 and i - prev_i > 490:
                    break
                i += 1
            i += 1
            if text[prev_i:i] == '':
                prev_i = i
            else:
                sentence_separated += [text[prev_i:i]]
                prev_i = i
        return self.clear_empty_elements(sentence_separated)

    def tokenize(self, sentences):
        """

        Parameters
        ----------
        sentences: list[str]

        Returns
        -------
        list[str]
            Token size ni 510-s doosh hemjeetei uguulberuud
        """
        listOfLen = []
        new_sentences = []
        queue = []
        for sentence in sentences:
            queue += [sentence]
            while len(queue) > 0:
                piece = queue.pop(0)
                size = len(self.fTokenizer.tokenize(piece))
                if size >= 510:
                    left, right = self.cut_around_middle(piece)
                    queue.append(left)
                    queue.append(right)
                else:
                    listOfLen += [size]
                    new_sentences += [piece]
        return listOfLen, new_sentences

    @staticmethod
    def cut_around_middle(string):
        """Uguulberiig dund hawiar ni taslah (space oldwol space-r)

        Parameters
        ----------
        string: str

        Returns
        -------
        str
            cut string
        """
        length = len(string)
        boundary_left = int(length * 0.45)
        boundary_right = int(length * 0.55) + 1
        for i in range(boundary_left, boundary_right):
            if not string[i].isalpha():
                return string[:i], string[i:]
        return string[:length // 2], string[length // 2:]

    @staticmethod
    def tagUniter(list_of_iob):
        """B bolon araas ni zalgagdah I tagtai NER ugnuudiig
        "_"-aar 1 buren ug bolgoh

        Parameters
        ----------
        list_of_iob: list[str,str]
            element=[word, tag_of_the_word] .i.e [['Trump', 'B-Person'],...]

        Returns
        -------
        list[str,str]
            .i.e [['Donald_Trump','Person'],...]
        """
        word = ''
        tag = ''
        unitedList = []
        for tagged in list_of_iob:
            if tagged[1][0] == 'B' and word != '':
                unitedList += [[word, tag]]
                word = ''
            if tagged[1][0] == 'I':
                word += '_'
            word += tagged[0]
            tag = tagged[1][2:]
        return unitedList + [[word, tag]]

    @staticmethod
    def from_list_to_texts(lis):
        """List-d baigaa stringiig 1 tom string ruu huwirgah

        Parameters
        ----------
        lis:[str]

        Returns
        -------
        str

        """
        texts = ""
        for string in lis:
            if str(string) == '-':
                texts = texts[:-1]
                texts += str(string)
            else:
                texts += str(string) + " "
        return texts

    @staticmethod
    def get_page_number(dic):
        return dic.get('page_number')

    @staticmethod
    def clear_empty_elements(_list):
        """Delete empty string inside input list

        Parameters
        ----------
        _list: list[str]

        Returns
        -------
        list[str]
        """
        new = []
        for element in _list:
            if element.isspace() or element == '':
                break
            new += [element]
        return new

    def cleaner(self, text):
        return text.replace('\xa0', ' ')
