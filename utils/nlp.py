from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from tqdm import tqdm


def twitter_preprocessor():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time',
                   'date', 'number'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis',
                  'censored'},
        all_caps_tag="wrap",
        fix_text=True,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons]
    ).pre_process_doc
    return preprocessor


def twitter_preprocess():
    preprocessor = twitter_preprocessor()

    def preprocess(name, dataset):
        desc = "PreProcessing dataset {}...".format(name)

        data = [preprocessor(x)
                for x in tqdm(dataset, desc=desc)]
        return data

    return preprocess
