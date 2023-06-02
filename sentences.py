class Sentences:
    sentence = ""
    cantonese = ""
    isTranslated = False;

    @classmethod
    def set_sentence(cls, new):
        cls.sentence = new

    @classmethod
    def get_sentence(cls):
        return cls.sentence

    @classmethod
    def set_cantonese(cls, new):
        cls.isTranslated = True;
        cls.cantonese = new

    @classmethod
    def get_cantonese(cls):
        return cls.cantonese

    @classmethod
    def is_translated(cls):
        return cls.isTranslated