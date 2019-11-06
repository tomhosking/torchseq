import bigjson, json


field_names = ['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers']

null_answer = 'No Answer Present.'

filtered_data = []

num_multi_ctxts = 0

with open('./data/msmarco/train_v2.1.json', 'r') as fp:
    data = json.load(fp)

    for ix, row in data['answers'].items():
        
        if row[0] != null_answer:
            # OK, valid answer. Find the correct pieces
            answer = row[0]
            question = data['query'][ix]

            contexts = [ctxt for ctxt in data['passages'][ix] if ctxt['is_selected'] != 0]
            if len(contexts) == 0:
                continue

            # if len(contexts) > 1:
            #     print('multi contexts selected')
            #     num_multi_ctxts += 1
            # context = contexts[0]['passage_text']

            
            for c in contexts:
                context = c['passage_text']
                a_loc = context.find(answer)

                if a_loc > -1 and '?' in question:
                    filtered_data.append({'a': answer, 'c': context, 'q': question, 'a_pos': a_loc})

print(len(filtered_data))
with open('./data/msmarco/train_v2.1_extractive.json', 'w') as fp:
    json.dump(filtered_data, fp)
