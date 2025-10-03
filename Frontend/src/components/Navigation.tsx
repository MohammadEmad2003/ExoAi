import { NavLink } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useState } from "react";
import {
  Home,
  Info,
  Brain,
  Play,
  BarChart3,
  Users,
  BookOpen,
  Telescope,
  Atom,
  Shield,
  MoreHorizontal,
  Menu,
  X,
  Upload,
} from "lucide-react";
import LanguageSwitcher from "./LanguageSwitcher";
import { Button } from "./ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import { useIsMobile } from "@/hooks/use-mobile";

const getPrimaryNavItems = (t: any) => [
  { to: "/", label: t("navigation.home"), icon: Home },
  { to: "/onboarding", label: t("navigation.guidedMode"), icon: BookOpen },
  { to: "/research", label: t("navigation.researchHub"), icon: Brain },
  { to: "/demo", label: t("navigation.demo"), icon: Play },
  { to: "/quantum", label: t("navigation.quantum"), icon: Atom },
  { to: "/adversarial", label: t("navigation.adversarial"), icon: Shield },
];

const getSecondaryNavItems = (t: any) => [
  {
    to: "/visualizations",
    label: t("navigation.visualizations"),
    icon: BarChart3,
  },
  { to: "/team", label: t("navigation.team"), icon: Users },
  { to: "/resources", label: t("navigation.resources"), icon: Info },
];

export function Navigation() {
  const { t } = useTranslation();
  const primaryNavItems = getPrimaryNavItems(t);
  const secondaryNavItems = getSecondaryNavItems(t);
  const isMobile = useIsMobile();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 border-b border-border/50 backdrop-blur-lg bg-background/80">
      <div className="w-full max-w-none px-6 sm:px-3 py-4">
        <div className="flex items-center justify-between gap-6">
          {/* Logo */}
          <div className="flex items-center space-x-3 flex-shrink-0">
            <div className="stellar-pulse">
              <Telescope className="h-8 w-8 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gradient-stellar">
                {t("branding.name", "Cosmic Analysts ExoAI")}
              </h1>
              <p className="text-xs text-muted-foreground hidden sm:block">
                {t("branding.tagline", "NASA Challenge 2025")}
              </p>
            </div>
          </div>

          {/* Desktop Navigation Links */}
          <div className="hidden lg:flex items-center space-x-2 flex-1 justify-end min-w-0 flex-wrap">
            {primaryNavItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `nav-link flex items-center space-x-2 px-2 py-2 rounded-md transition-colors whitespace-nowrap ${
                      isActive ? "active bg-primary/10" : "hover:bg-primary/5"
                    }`
                  }
                >
                  <Icon className="h-4 w-4" />
                  <span className="text-sm font-medium">{item.label}</span>
                </NavLink>
              );
            })}

            {/* More dropdown for secondary items */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="nav-link flex items-center space-x-2 px-2 py-2 rounded-md transition-colors whitespace-nowrap gap-2"
                >
                  <MoreHorizontal className="h-4 w-4" />
                  <span className="text-sm font-medium">
                    {t("common.more")}
                  </span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="end"
                className="w-48 bg-card/95 backdrop-blur-sm z-[100]"
              >
                {secondaryNavItems.map((item) => {
                  const Icon = item.icon;
                  return (
                    <DropdownMenuItem key={item.to} asChild>
                      <NavLink
                        to={item.to}
                        className="flex items-center gap-2 cursor-pointer w-full"
                      >
                        <Icon className="h-4 w-4" />
                        <span>{item.label}</span>
                      </NavLink>
                    </DropdownMenuItem>
                  );
                })}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Language Switcher - Always visible on desktop */}
            <div className="ml-2">
              <LanguageSwitcher />
            </div>
          </div>

          {/* Mobile/Tablet menu button and language switcher */}
          <div className="lg:hidden flex items-center gap-2">
            <LanguageSwitcher />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </Button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="lg:hidden mt-4 pb-4 space-y-2 border-t border-border/50 pt-4">
            {[...primaryNavItems, ...secondaryNavItems].map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  onClick={() => setMobileMenuOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center space-x-3 px-4 py-3 rounded-md transition-colors ${
                      isActive
                        ? "bg-primary/20 text-primary"
                        : "hover:bg-primary/10"
                    }`
                  }
                >
                  <Icon className="h-5 w-5" />
                  <span className="font-medium">{item.label}</span>
                </NavLink>
              );
            })}
          </div>
        )}
      </div>
    </nav>
  );
}
