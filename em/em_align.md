# Вывод `WordAligner`

Выкладки несложные, но основная задача — не запутаться в трех уровнях индексации. Будем придерживаться следующих обозначений:

> * $l$ — число предложений в корпусе, дла них используем индекс $i\in\overline{1,l}$;
> * $n_i$ — число слов в $i$-м исходном предложении $s^i$, индекс по словам — $j\in\overline{1,n_i}$;
> * $m_i$ — число слов в $i$-м целевом предложении $t^i$, индекс по словам — $k\in\overline{1,m_i}$;
> 
> Например, $s^i_j$ и $t^i_k$ — это $j$-е слово $i$-го предложения на исходном языке и $k$-е слово его перевода соответственно.
> 
> * $a^i_k$ — латентная переменная, т.е. выравнивание $k$-го слова $i$-го целевого предложения ($t^i_k$);
> * $A=[a^1,\dots,a^l]$, $S=[s^1,\dots,s^l]$, $T=[t^1,\dots,t^l]$ — латентные переменные (выравнивания), исходные и целевые предложения по всему корпусу;
> * $\theta_{xy}:= \theta(y\mid x)$, $x\in\mathcal{X}$ — множество слов на исходном языке, $y\in\mathcal{Y}$ — множество слов на целевом языке (понадобится на M-шаге).

Тогда, согласно нашей модели:

$$
p(a^i_k=j, t^i_k \mid s^i)=
p(a^i_k=j)\cdot p(t^i_k\mid a^i_k=j, s^i)=
\frac{\theta(t^i_k\mid s^i_j)}{n_i}.
$$

**E-шаг.** Здесь мы обновляем распределение скрытых переменных $q^*(A)$, полагая его равным апостериорному распределению $p(A\mid T,S)$:

$$
q_{ijk}:=
q^*(a^i_k=j)=
p(a^i_k=j\mid t^i_k, s^i)=
\frac{p(a^i_k=j, t^i_k\mid s^i)}{p(t^i_k\mid s^i)}
= \\ =
\frac{p(a^i_k=j, t^i_k\mid s^i)}
{\sum\limits_{\ddot{\jmath}=1}^{n_i}
p(a^i_k=\ddot{\jmath}, t^i_k\mid s^i)}
=
\frac{\frac{1}{n_i}\cdot\theta(t^i_k\mid s^i_j)}
{\sum\limits_{\ddot{\jmath}=1}^{n_i}
\frac{1}{n_i}\cdot\theta(t^i_k\mid s^i_{\ddot{\jmath}})}
= \\ =
\frac{\theta(t^i_k\mid s^i_j)}
{\sum\limits_{\ddot{\jmath}=1}^{n_i}
\theta(t^i_k\mid s^i_{\ddot{\jmath}})}.
$$

**Нижняя оценка логарифма полного правдоподобия $\mathcal{L}$.** 

$$
\mathcal{L}=\int q^*(A)\cdot\log \frac{p(T, A)}{q^*(A)}
=\\=
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot\log
\frac{p(a^i_k=j, t_k\mid s^i)}{q_{ijk}}
=\\=
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot\log
\frac{\theta(t^i_k\mid s^i_k)}{n_i\cdot q_{ijk}}.
$$

**M-шаг**. Теперь мы хотим обновить значения переменных $\theta(y\mid x)=\theta_{xy}$, максимизируя нижнюю оценку логарифма правдоподобия — $\mathcal{L}$.

Отбросим все лишнее:
$$
\mathcal{L}=
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot\log
\frac{\theta(t^i_k\mid s^i_k)}{n_i\cdot q_{ijk}}
\ \rightarrow\ \max_{\theta} \\
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot\log {\theta(t^i_k\mid s^i_k)}
\ \rightarrow\ \max_{\theta}
$$

У нас задача максимизации с ограничениями,
$$
\forall x\in\mathcal{X}\quad \sum\limits_{y\in\mathcal{Y}} \theta_{xy}=1,
$$
поэтому будем дифференцировать лагранжиан $\mathfrak{L}$ со множителями $\lambda_x$ для каждого равенства:

$$
\mathfrak{L}=
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot\log {\theta(t^i_k\mid s^i_k)}
-
\sum\limits_{x\in\mathcal{X}}
\lambda_x\cdot
\left(\sum\limits_{y\in\mathcal{Y}} \theta_{xy}-1\right),
$$

Продифференцируем по $\theta_{xy}$:
$$
\frac{\partial\mathfrak{L}}{\partial\theta_{xy}}
=
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot
\frac{\partial}{\partial\theta_{xy}}
\log {\theta(t^i_k\mid s^i_k)}
-
\frac{\partial}{\partial\theta_{xy}}
\sum\limits_{x\in\mathcal{X}}
\lambda_x\cdot
\left(\sum\limits_{y\in\mathcal{Y}} \theta_{xy}-1\right)
= \\ =
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot
\frac{1}{\theta_{xy}}\cdot[s^i_j=x]\cdot[t^i_k=y]-\lambda_x.
$$

Приравнивая производную нулю, выражаем $\theta_{xy}$:
$$
\theta_{xy}=\frac{1}{\lambda_{x}}
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot
[s^i_j=x]\cdot[t^i_k=y],
$$

суммируем по всем $y\in\mathcal{Y}$:

$$
\sum\limits_{y\in\mathcal{Y}} \theta_{xy}=1
=\\=
\sum\limits_{y\in\mathcal{Y}}
\frac{1}{\lambda_{x}}
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot
[s^i_j=x]\cdot[t^i_k=y]
=\\=
\frac{1}{\lambda_{x}}
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot
[s^i_j=x]\cdot
\underbrace{\sum\limits_{y\in\mathcal{Y}}[t^i_k=y]}_{1}.
$$

Можем выразить множители Лагранжа:
$$
\lambda_x=
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}
[s^i_j=x].
$$

Осталось подставить в подставить в выражение для $\theta_{xy}$:
$$
\theta_{xy}=\frac
{\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot
[s^i_j=x]\cdot[t^i_k=y]}
{\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}
[s^i_j=x]}.
$$

Почему на полученных значениях действительно достигается максимум? Потому что $\theta_{xy}$ входят в $\mathcal{L}$ только через логарифм, т.е. вогнутую функцию, а все наложенные ограничения — линейные. Таким образом, выполняется условие Слейтера (с заменой минимума на максимум и выпуклости на вогнутость).

# Вывод `WordPositionAligner`

Сохраним обозначения с вывода для `WordAligner`, но добавим новое для  $\phi$:

> $\phi^{n_i m_i}_{jk} := \phi_{m_i,n_i}(j\mid k)$ — это проще согласуется с размером и индексацией;

Отличие от прошлого вывода — замена $\frac{1}{n_i}$ на набор $\phi^{n_i m_i}_{jk}$. Таким образом, модель приобретает вид

$$
p(a^i_k=j, t^i_k \mid s^i)=
p(a^i_k=j\mid m_i, n_i)\cdot p(t^i_k\mid a^i_k=j, s^i)=
\phi^{n_i m_i}_{jk}\cdot\theta(t^i_k\mid s^i_j).
$$

**E-шаг**. Здесь пока все хорошо, просто подменяем:

$$
q_{ijk}:=
q^*(a^i_k=j)=
p(a^i_k=j\mid t^i_k, s^i)=
\frac{p(a^i_k=j, t^i_k\mid s^i)}{p(t^i_k\mid s^i)}
= \\ =
\frac{p(a^i_k=j, t^i_k\mid s^i)}
{\sum\limits_{\ddot{\jmath}=1}^{n_i}
p(a^i_k=\ddot{\jmath}, t^i_k\mid s^i)}
=
\frac
{\phi^{n_i m_i}_{jk}\cdot\theta(t^i_k\mid s^i_j)}
{\sum\limits_{\ddot{\jmath}=1}^{n_i}
\phi^{n_i m_i}_{{\ddot{\jmath}}k}\cdot\theta(t^i_k\mid s^i_{\ddot{\jmath}})
}.
$$

**Нижняя оценка логарифма полного правдоподобия $\mathcal{L}$.** Здесь пока тоже без сюрпризов!

$$
\mathcal{L}=\int q^*(A)\cdot\log \frac{p(T, A)}{q^*(A)}
=\\=
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot\log
\frac{p(a^i_k=j, t_k\mid s^i)}{q_{ijk}}
=\\=
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot\log
\frac{\phi^{n_i m_i}_{jk}\cdot\theta(t^i_k\mid s^i_j)}{q_{ijk}}.
$$

**M-шаг**. А вот сейчас начнется...

Во-первых, заметим, что логарифм факторизует $\mathcal{L}$ на отдельные компоненты с $\theta$ и $\phi$. Причем часть с $\theta$ полностью совпадает с предыдущей моделью, поэтому сразу выпишем значения для аналитического максимума:

$$
\theta_{xy}=\frac
{\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\cdot
[s^i_j=x]\cdot[t^i_k=y]}
{\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}
[s^i_j=x]}.
$$

Теперь разберемся с $\phi$. Мы максимизируем 
$$
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\log {\phi^{n_i m_i}_{jk}} \ \rightarrow \ \max_{\phi},
$$

причем на $\phi$ имеются ограничения — нормировка вероятностей:
$$
\forall i\in\overline{1,l}, \forall k\in \overline{1,m_i} \quad
\sum\limits_{j=1}^{n_i} \phi^{n_i m_i}_{jk} =1.
$$

Записываем лагранжиан $\mathfrak{L}$:
$$
\mathfrak{L}=
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\log {\phi^{n_i m_i}_{jk}} -
\sum\limits_{i=1}^{l}
\sum\limits_{k=1}^{m_i}
\lambda_{ik}
\sum\limits_{j=1}^{n_i}\left(\phi^{n_i m_i}_{jk} -1\right),
$$

дифференцируем по $\phi^{n_a m_a}_{bc}$ (т.е. $i,j,k$ соответствуют $a,b,c$):
$$
0=\frac{\partial{\mathfrak{L}}}{\partial{\phi^{n_a m_a}_{bc}}}=
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}\frac{1}{\phi^{n_i m_i}_{jk}}
[n_i,m_i,j,k=n_a,m_a,b,c]
-
\sum\limits_{i=1}^{l}
\sum\limits_{k=1}^{m_i}
\lambda_{ik}[n_i,m_i,k=n_a,m_a,c],
$$

где равенство внутри индикаторов означает проверку того, каждый элемент левой части равен соответствующему элементу правой. Выражаем $\phi^{n_a m_a}_{bc}$:

$$
\phi^{n_a m_a}_{bc}=
\frac
{
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}
[n_i,m_i,j,k=n_a,m_a,b,c]
}
{
\sum\limits_{i=1}^{l}
\sum\limits_{k=1}^{m_i}
\lambda_{ik}[n_i,m_i,k=n_a,m_a,c]
}.
$$

Суммируем обе части равенства по $b\in\overline{1,n_a}$ и учитываем ограничения:
$$
\sum\limits_{b=1}^{n_a}
\phi^{n_a m_a}_{bc}
=1
= \\ =
\frac
{
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}
[n_i,m_i,j,k=n_a,m_a,b,c]
}
{
\sum\limits_{i=1}^{l}
\sum\limits_{k=1}^{m_i}
\lambda_{ik}[n_i,m_i,k=n_a,m_a,c]
}
= \\ =
\frac
{
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}
\sum\limits_{b=1}^{n_a}
[n_i,m_i,j,k=n_a,m_a,b,c]
}
{
\sum\limits_{i=1}^{l}
\sum\limits_{k=1}^{m_i}
\lambda_{ik}[n_i,m_i,k=n_a,m_a,c]
}
= \\ =
\frac
{
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}
[n_i,m_i,k=n_a,m_a,c]
}
{
\sum\limits_{i=1}^{l}
\sum\limits_{k=1}^{m_i}
\lambda_{ik}[n_i,m_i,k=n_a,m_a,c]
}
\ \implies \\
\sum\limits_{i=1}^{l}
\sum\limits_{k=1}^{m_i}
\lambda_{ik}[n_i,m_i,k=n_a,m_a,c]
=
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}
[n_i,m_i,k=n_a,m_a,c]
$$

Ой, а это же как раз знаменатель, который нам нужен в ранее полученном выражении для $\phi^{n_a m_a}_{bc}$. Подставляем и получаем искомые значения:

$$
\phi^{n_a m_a}_{bc}=
\frac
{
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}
[n_i,m_i,j,k=n_a,m_a,b,c]
}
{
\sum\limits_{i=1}^{l}
\sum\limits_{j=1}^{n_i}
\sum\limits_{k=1}^{m_i}
q_{ijk}
[n_i,m_i,k=n_a,m_a,c]
}.
$$