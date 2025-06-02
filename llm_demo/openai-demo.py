#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""

"""

from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
