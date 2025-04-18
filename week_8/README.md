# CU OmutAI
---
## Week 8. SAE
**Цель:**
    В этом задании вам нужно будет научиться изменять поведение моделей с помощью SAE.

**Постановка**
Пусть декодер $W_{dec} \in \R^{F \times d}$ состоит из строк $e_{i} \in \R^d$, а $\bold h \in \R^d$ – хиден модели. 


#### Способ 1 (Default)
Характекризуется следующей формулой:
$\tilde{h} = h + \alpha \cdot e_{i}, \ \alpha \in \R^{+}$     

#### Способ 2 (Normalized)
Характекризуется следующей формулой:
$\tilde{h} = \frac{h + \alpha \cdot e_{i}}{||h + \alpha \cdot e_{i}||} \cdot ||\bold h||, \ \alpha \in \R^{+}$    

### Результаты
В качестве модели, которую стирим, возьмем [*google/gemma-2-9b-it*](https://huggingface.co/google/gemma-2-9b-it). 
Причиной выбора более большой версии модели стало отсутствие у *2b* версии нормальных весов SAE, а только через библиотеку *pyvene*. 

В качестве фичи для стирки использовалась фича про **[котиков](https://www.neuronpedia.org/gemma-2-9b-it/20-gemmascope-res-131k/57268)**.

LLM для оценки текста: [*Qwen/Qwen2.5-7B-Instruct*](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

Промпт для оценки fluency текста и behavioral score был взят [статьи](https://arxiv.org/pdf/2411.02193).

> BEHAVIORAL CRITERION = 'The text contains discussion about a cats.'

> FLUENCY CRITERION = 'Correctly and coherently constructed sentences, do not contain grammatical errors.'


Диапазон $\alpha$ был взят одинаковый
 $\alpha = \{0, 10,...,140, 150\}$, $|\alpha|=16$.

Ниже представлены графики получившихся результатов.

<figure>
<img src="./assets/behavioral.png"></img>
</figure>

<figure>
<img src="./assets/fluency.png"></img>
</figure>

#### Вывод:
На определенных значениях $\alpha$ нам удалось достаточно хорошо застирить модель, но при этом потеряв во fluency текста. 
Так, например, при $\alpha=100$, у нас получились следующие тексты:
**Default**
> I am a large language model, trained by the Gemma team. I am a relatively new model, but I am already quite popular.
I am a very curious cat, and I love to explore. I am also a very affectionate cat, and I love to cuddle.
I am a very independent cat, and I am a very good hunter. I am also a very good mouser.
I am a very good companion, and I am a very good friend.
I am a very good cat.

**Normalized**

