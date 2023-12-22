import os
import platform
import ann
import rnn

def clear_screen():
    system = platform.system()
    if system == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def main():
    print("Yapay Zeka Oluşturucusuna Hoş Geldiniz.")

    operating_system = input("Öncelikle İşletim Sisteminizi Öğrenebilir miyim? (windows/linux): ").lower()
    clear_screen()

    if operating_system not in ["windows", "linux"]:
        print("Hatalı işletim sistemi girdisi. Sadece 'windows' veya 'linux' giriniz.")
        return

    choice = input("Ne Tür Bir Yapay Zeka Yapmak İstersiniz (ANN, RNN, CNN): ").lower()

    if choice == "ann":
        create_ann()
    elif choice == "rnn":
        create_rnn()
    elif choice == "cnn":
        create_cnn()
    else:
        print("Geçersiz bir seçenek girdiniz. Lütfen 'ANN', 'RNN' veya 'CNN' seçin.")

def create_ann():
    print("Senden Birkaç Parametre Alacağım")
    shape = int(input("Shape (örn: 1,.., zorunlu katman): ") or 10)
    units = int(input("Units: ") or 10)
    activation = input("Activation (varsayılan: None): ").strip() or None
    use_bias = input("Use Bias (True/False) (varsayılan: False): ").strip().lower() == 'true'
    kernel_initializer = input("Kernel İnitializer (varsayılan: glorot_uniform): ").strip() or "glorot_uniform"
    bias_initializer = input("Bias İnitializer (varsayılan: zeros): ").strip() or "zeros"
    out_activation = input("Output Activation (varsayılan: relu): ").strip() or "relu"

    ann.ann(shape=shape,units=units, activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            out_activation=out_activation)

def create_rnn():
    print("Senden Birkaç Parametre Alacağım")
    input_dim = int(input("Input Dim (zorunlu katman): ") or 10)
    output_dim = int(input("Output Dim (zorunlu katman): ") or 10)
    input_length = int(input("Input Length (varsayılan: None): ") or None)

    units = int(input("Units (zorunlu katman): ") or 10)
    activation = input("Activation (varsayılan: relu): ").strip() or "relu"
    use_bias = input("Use Bias (t/f) (varsayılan: True): ").strip().lower() == 't'
    kernel_initializer = input("Kernel İnitializer (varsayılan: glorot_uniform): ").strip() or "glorot_uniform"
    bias_initializer = input("Bias İnitializer (varsayılan: zeros): ").strip() or "zeros"
    drop_out = float(input("Drop Out (float: varsayılan: 0.0): ") or 0.0)

    out_activation = input("Output Activation (varsayılan: relu): ").strip() or "relu"

    rnn.rnn(input_dim, output_dim, input_length, units, activation, use_bias,
            kernel_initializer, bias_initializer, drop_out, out_activation)

def create_cnn():
    print("dd")

if __name__ == "__main__":
    main()
