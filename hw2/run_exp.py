import os


def exp1():
    for K in [32, 64]:
        for L in [2, 4, 8, 16]:
            os.system(
                "python -m hw2.experiments run-exp -n exp1_1 -K {0} -L {1} -P 3 -H 500 500 --early-stopping 10".format(
                    K, L
                )
            )


def exp2():
    for K in [32, 64, 128]:
        for L in [2, 4, 8]:
            os.system(
                "python -m hw2.experiments run-exp -n exp1_2 -K {0} -L {1} -P 3 -H 500 500 --early-stopping 10".format(
                    K, L
                )
            )


def exp3():
    for L in [2, 3, 4]:
        os.system(
            "python -m hw2.experiments run-exp -n exp1_3 -K 64 128 -L {0} -P 3 -H 500 500 --early-stopping 10".format(
                L
            )
        )


def exp4():
    for L in [8, 16, 32]:
        os.system(
            "python -m hw2.experiments run-exp -n exp1_4 -K 32 -L {0} -P 6 -H 500 500 -M resnet --early-stopping 10".format(
                L
            )
        )

    for L in [2, 4, 8]:
        os.system(
            "python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L {0} -P 7 -H 500 500 -M resnet --early-stopping 10".format(
                L
            )
        )


def main():
    # exp1()
    # exp2()
    # exp3()
    exp4()


if __name__ == "__main__":
    main()
