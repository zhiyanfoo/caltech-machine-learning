def main():
    output(ans)

def output(ans):
    print("no simulations required for this weeks assignment")
    print("displaying mutiple choice answers instead")
    for key in sorted(ans.keys()):
        print("""question""", key, ":", ans[key])

ans = {
        1 : 'b',
        2 : 'c',
        3 : 'd',
        4 : 'aXb',
        5 : 'eXb',
        6 : 'c',
        7 : 'e',
        8 : 'd',
        9 : 'aXd',
        10 : 'b',
        }

if __name__ == '__main__':
    main()
