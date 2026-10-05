# Troubleshooting

## Problems accessing Azure

If Azure deployment fails with an AuthorizationFailed error, one possible cause is that the wrong Azure subscription is selected.

Check the current default subscription:

```text
az account show --output table
```

To list available subscriptions:

```text
az account list --output table
```

To change the default subscription:

```text
az account set -s <subscription-name-or-id>
```

Then retry the deployment. You must have permission to create resources in the selected subscription/resource group.

## `csshx` fails on macOS

On macOS, `csshx` can be installed using Homebrew:

```bash
brew install csshx
```

On some versions of macOS, `csshx` may fail with an error beginning with:

```text
Unimplemented: POSIX::tmpnam()
```

This appears to result from an incompatibility between `csshx` and newer versions of Perl. One workaround is to install a fork of `csshx` that addresses this issue:

```bash
brew tap parera10/csshx
brew install parera10/csshx/csshx
```

Alternatively, the issue can be fixed by replacing the following line in the `csshx` script:

```perl
use POSIX qw(tmpnam);
```

with:

```perl
use File::Temp qw/ :POSIX /;
```

See the [csshx GitHub issue](https://github.com/brockgr/csshx/issues/103) for additional information.

## Problem with export-to-json if there is a space in the path

If you try to export-to-json to a location with a space in the path `~/OneDrive\ -\ Princeton\ University/million-monkeys-results-shared/v0.2.6-comprep-lev1-run2` it will fail. The simplest solution is find a place with no space in the path.

## Permission denied (publickey)

If the VMs say "permission denied (publickey)", make sure that in your `~/.ssh/config` file on your local machine, you have written:

*IdentityFile id_rsa*

Not:

*IdentityFile id_rsa.pub*