# Copyright 2020 The HuggingFace Datasets Authors and the current
# dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""The SpamAssassin public mail corpus"""


import email
import email.policy
import codecs
import json
import urllib.parse

import datasets
from .dep import ftfy, wcwidth


# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
Welcome to the SpamAssassin public mail corpus.  This is a selection of mail
messages, suitable for use in testing spam filtering systems.  Pertinent
points:

  - All headers are reproduced in full.  Some address obfuscation has taken
    place, and hostnames in some cases have been replaced with
    "spamassassin.taint.org" (which has a valid MX record).  In most cases
    though, the headers appear as they were received.

  - All of these messages were posted to public fora, were sent to me in the
    knowledge that they may be made public, were sent by me, or originated as
    newsletters from public news web sites.

  - relying on data from public networked blacklists like DNSBLs, Razor, DCC
    or Pyzor for identification of these messages is not recommended, as a
    previous downloader of this corpus might have reported them!

  - Copyright for the text in the messages remains with the original senders.


OK, now onto the corpus description.  It's split into three parts, as follows:

  - spam: 500 spam messages, all received from non-spam-trap sources.

  - easy_ham: 2500 non-spam messages.  These are typically quite easy to
    differentiate from spam, since they frequently do not contain any spammish
    signatures (like HTML etc).

  - hard_ham: 250 non-spam messages which are closer in many respects to
    typical spam: use of HTML, unusual HTML markup, coloured text,
    "spammish-sounding" phrases etc.

  - easy_ham_2: 1400 non-spam messages.  A more recent addition to the set.

  - spam_2: 1397 spam messages.  Again, more recent.

Total count: 6047 messages, with about a 31% spam ratio.
"""

_HOMEPAGE = "https://spamassassin.apache.org/old/publiccorpus/readme.html"

_FILES = [
    "20021010_easy_ham.tar.bz2",
    "20021010_hard_ham.tar.bz2",
    "20021010_spam.tar.bz2",
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20030228_spam_2.tar.bz2",
    "20050311_spam_2.tar.bz2",
]


class MessageParser:
    def __init__(self):
        self.policy = email.policy.default.clone(
            utf8=True,
            refold_source='none')

        def get_text(payload, charset):
            try:
                text = codecs.decode(payload, charset)
                return ftfy.fix_encoding(text)
            except UnicodeDecodeError:
                pass
            except LookupError:
                pass
            text, charset = ftfy.guess_bytes(payload)
            return text

        self.get_text = get_text

    def pick(self, msg):
        # TODO: it might be worthwhile to include headers. They are
        # certainly informative, but difficult to scrub of artifacts
        # that would not generalize well.
        if msg.is_multipart():
            return [self.pick(part) for part in msg.get_payload()]
        ct = msg.get_content_type()
        if ct[:5] == "text/":
            payload = msg.get_payload(decode=True)
            charset = msg.get_param("charset", "utf-8")
            return self.get_text(payload, charset)
        return "…"

    def __call__(self, raw):
        if b"Message-Id: <>" in raw:
            # email.message seems to explode on MsgId "<>"
            return None
        msg = email.message_from_bytes(raw, policy=self.policy)
        obj = self.pick(msg)
        return json.dumps(obj, ensure_ascii=False)


class SpamAssassin(datasets.GeneratorBasedBuilder):
    """SpamAssassin public mail corpus"""

    VERSION = datasets.Version("0.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="text",
            version=VERSION,
            description="Flattened mime data and normalized character sets",
        ),
        datasets.BuilderConfig(
            name="unprocessed",
            version=VERSION,
            description="Raw original input files in binary",
        ),
    ]

    DEFAULT_CONFIG_NAME = "text"

    def _info(self):
        if self.config.name == "unprocessed":
            features = {'raw': datasets.Value(dtype='binary')}
        else:
            features = {'text': datasets.Value(dtype='string')}
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'label': datasets.ClassLabel(
                    num_classes=2,
                    names=['spam', 'ham']),
                'group': datasets.Value(dtype='string'),
                **features
            }),
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        srcs = [urllib.parse.urljoin(_HOMEPAGE, file) for file in _FILES]
        srcs = [dl_manager.download(url) for url in srcs]
        srcs = [dl_manager.iter_archive(path) for path in srcs]
        return [datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={"srcs": srcs}
        )]

    def _extract_tars(self, src):
        for arch in src:
            for name, fh in arch:
                group = name.split('/')[0]
                label = 'ham' if 'ham' in group else 'spam'
                yield dict(label=label, group=group, raw=fh.read())

    def _parse_messages(self, src):
        parser = MessageParser()
        for row in src:
            text = parser(row["raw"])
            if text is not None:
                yield dict(label=row["label"], group=row["group"], text=text)

    def _generate_examples(self, srcs):
        gen = self._extract_tars(srcs)
        if self.config.name == "text":
            gen = self._parse_messages(gen)
        yield from enumerate(gen)
