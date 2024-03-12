import numpy as np
import matplotlib.pyplot as plt


def get_best_answer(start_logit, end_logit, offsets, context, example_id):
    n_best = 20
    max_answer_length = 30
    predicted_answers = []
    answers = []
    # Loop through all features associated with that example

    start_indexes = np.argsort(start_logit)[-1:-n_best - 1:-1].tolist()
    end_indexes = np.argsort(end_logit)[-1:-n_best - 1:-1].tolist()

    for start_index in start_indexes:
        for end_index in end_indexes:

            mask = np.zeros_like(start_logit)
            mask[start_index] = 1
            mask[end_index] = 2

            # Skip answers that are not fully in the context
            if offsets[start_index] is None or offsets[end_index] is None:
                continue
            # Skip answers with a length that is either < 0 or > max_answer_length
            if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
            ):
                continue

            answer = {
                "text": context[offsets[start_index][0]: offsets[end_index][1]],
                "logit_score": start_logit[start_index] + end_logit[end_index],
                "mask": mask
            }
            answers.append(answer)

    # Select the answer with the best score
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["logit_score"])
        predicted_answer = {"id": example_id, "prediction_text": best_answer["text"], "mask": best_answer['mask']}

    else:
        predicted_answer = {"id": example_id, "prediction_text": "", "mask": None}

    return predicted_answer


def plot_conservation(Ls, Rs, filename=None):
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(Ls, Rs, s=20, label='R', c='red')
    ax.plot(Ls, Ls, color='black', linestyle='-', linewidth=1)

    # ax.set_ylabel('$\sum_i R_i$', fontsize=30, usetex=True)
    # ax.set_xlabel('output $f$', fontsize=30,  usetex=True)
    ax.set_ylabel('$\sum_i R_i$', fontsize=30, usetex=False)
    ax.set_xlabel('output $f$', fontsize=30, usetex=False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)
    f.tight_layout()
    if filename:
        f.savefig(filename, dpi=100)
        plt.close()
    else:
        plt.show()
